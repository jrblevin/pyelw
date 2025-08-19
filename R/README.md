This directory contains R code implementing various local Whittle estimators
originally from the LongMemoryTS package by Christian Leschinski, Michelle
Voges, and Kai Wenger. This package was retired from CRAN in July of 2025,
however, we provide here updated versions of `local_W.R` and
`ELW_est.R` with a corrected implementation of the Hurvich and
Chen (2000) tapered LW estimator and the two-step exact local Whittle
estimator of Shimotsu (2010).  Details on the changes can be found in
the PyELW paper:

* Blevins, J.R. (2025).
  [PyELW: Exact Local Whittle Estimation for Long Memory Time Series in Python](https://jblevins.org/research/pyelw).
  Working Paper, The Ohio State University.

For convenience, here are the differences in `local_W.R`:

```diff
--- R/local_W_bad.R
+++ R/local_W.R
@@ -33,8 +33,18 @@

 #'Complex Cosine Bell Taper
 #'@keywords internal
-cos_bell_cmplx<-function(u){1/2*(1-cos(1i*2*pi*u))}
+cos_bell_cmplx<-function(u){1/2*(1-exp(1i*2*pi*u))}

+#'Tapered periodogram of Hurvich and Chen (2000)
+#'@keywords internal
+hc_per<-function(data, m){
+  T <- length(data)
+  norm_factor <- sqrt(2 * pi * T) * sqrt(2)
+  lambda <- 2 * pi * seq_len(m) %o% seq_len(T) / T
+  W <- rowSums(exp(1i * lambda) * rep(data, each = m)) / norm_factor
+  return(abs(W)^2)
+}
+
 #'Concentrated local Whittle likelihood for tapered estimate. Only for internal use. Cf. Velasco (1999).
 #'@keywords internal
 R.lw.tapered<-function(d,peri,m,p,T){
@@ -104,10 +114,10 @@
   if(taper=="HC"){
     data<-diff(data,differences=diff_param)
     T<-length(data)
-    ht<-cos_bell_cmplx((1:T)/T)
+    ht<-cos_bell_cmplx((1:T-0.5)/T)
     data<-ht*data
-    peri<-per(data)[-1]
-    d.hat<-optimize(f=R.lw.hc, interval=int, peri=peri,  m=m, T=T)$minimum+1
+    peri <- hc_per(data, m) # HC tapered periodogram
+    d.hat<-optimize(f=R.lw.hc, interval=int, peri=peri,  m=m, T=T)$minimum+diff_param
     se<-sqrt(1.5/(4*m))
   }
   if(taper=="none"){
```

Similarly, here are the differences in `ELW_est.R`:

```diff
--- ELW_est_bad.R
+++ ELW_est.R
@@ -8,7 +8,15 @@

 #' Concentrated local Whittle likelihood. Only for internal use. cf. Shimotsu and Phillips (2005), p. ???.
 #'@keywords internal
-wd.elw<-function(d){1/2*(1+cos(4*pi*d))}
+wd.elw<-function(d){
+  if(d <= 0.5) {
+    return(1.0)
+  } else if(d < 0.75) {
+    return(0.5 * (1 + cos(4*pi*d)))
+  } else {
+    return(0.0)
+  }
+}

 #' concentrated likelihood function for ELW estimator
 #'@keywords internal
@@ -23,10 +31,12 @@
 #' concentrated likelihood function for ELW estimator - weighted version
 #'@keywords internal
 R.elw.weighted<-function(d,data,m){
-  data<-(data-wd.elw(d)*mean(data)-(1-wd.elw(d))*data[1])[-1]
-  T<-length(data)
+  weight<-wd.elw(d)
+  myu<-weight*mean(data) + (1 - weight)*data[1]
+  data_corrected<-data-myu
+  T<-length(data_corrected)
   lambda<-2*pi/T
-  Peri<-per(fdiff(data, d=d))[-1]
+  Peri<-per(fdiff(data_corrected, d=d))[-1]
   K<-log(mean(Peri[1:m]))-2*d*mean(log(lambda*(1:m)))
   K
 }
@@ -103,16 +113,19 @@
 #' @export
 ELW2S<-function(data, m, trend_order=0, taper=c("Velasco","HC")){
   taper<-taper[1]
-  aux_est<-local.W(data=data, m=m, taper=taper, int=c(-1/2,2.5))
-  d_init<-aux_est$d
   T<-length(data)
   if(trend_order==0){
-    Xt<-residuals(lm(data~1))}else{
+    Xt<-residuals(lm(data~1))
+  }else{
       time<-1:T
       Xt<-residuals(lm(data~poly(time,trend_order)))
     }
-  d.hat<-optim(par=d_init, fn=R.elw.weighted, method="BFGS", data=data, m=m)$par
-  if(abs(d.hat-d_init)>2.5*aux_est$s.e.){d.hat<-d_init}
+  aux_est<-local.W(data=Xt, m=m, taper=taper, int=c(-1/2,2.5))
+  d_init<-aux_est$d
+  d.hat<-optim(par=d_init, fn=R.elw.weighted, method="BFGS", data=Xt, m=m)$par
+  if(abs(d.hat-d_init)>2.5*aux_est$s.e.){
+    d.hat<-d_init
+  }
   se<-1/(2*sqrt(m))
   return(list("d"=d.hat, "s.e."=se))
 }
```

The LongMemoryTS package was released under the GPL-2 license and contained the
following `DESCRIPTION` file:

```
Package: LongMemoryTS
Type: Package
Title: Long Memory Time Series
Version: 0.1.0
Date: 2019-01-18
Authors@R: c(person("Christian", "Leschinski", email="christian_leschinski@gmx.de", role=c("aut", "cre")), person("Michelle", "Voges", email="voges@statistik.uni-hannover.de", role="ctb"), person("Kai", "Wenger", email="wenger@statistik.uni-hannover.de", role="ctb"))
Description: Long Memory Time Series is a collection of functions for estimation, simulation and testing of long memory processes, spurious long memory processes and fractionally cointegrated systems. 
License: GPL-2
LinkingTo: Rcpp, RcppArmadillo
Imports: Rcpp, stats, longmemo, partitions, fracdiff, mvtnorm
RoxygenNote: 6.1.1
NeedsCompilation: yes
Packaged: 2019-02-09 13:31:27 UTC; Christian
Author: Christian Leschinski [aut, cre],
  Michelle Voges [ctb],
  Kai Wenger [ctb]
Maintainer: Christian Leschinski <christian_leschinski@gmx.de>
Repository: CRAN
Date/Publication: 2019-02-18 14:40:03 UTC
```
