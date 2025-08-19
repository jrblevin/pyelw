// ssc install whittle

log using stata_whittle.log

clear all
insheet using ../data/nile.csv
tsset year, yearly
whittle nile, powers(0.4 0.5 0.6 0.7 0.8 0.9)
whittle nile, powers(0.4 0.5 0.6 0.7 0.8 0.9) exact

clear all
insheet using ../data/sealevel.csv
gen date = monthly(datevec, "YM")
format date %tm
tsset date
whittle sea, powers(0.6 0.65 0.7 0.75 0.8)
whittle sea, powers(0.6 0.65 0.7 0.75 0.8) exact

log close
