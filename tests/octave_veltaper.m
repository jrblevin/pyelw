% Generates test cases using the tapered local Whittle estimator of
% Velasco (1999) with Shimotsu's implementation of the Kolmogorov
% taper (p=3). Requires Shimotsu's veltaper.m function.

printf('GNU Octave version %s\n', version());

% Load datasets directly from CSV files
% Skip header row and load specific columns
nile_full = csvread('../data/nile.csv', 1, 0);
sealevel_full = csvread('../data/sealevel.csv', 1, 0);
nile = nile_full(:, 2);
sealevel = sealevel_full(:, 2);

printf('Loaded nile (%d obs) and sealevel (%d obs)\n', length(nile), length(sealevel));

% Test configurations
alpha_values = [0.55, 0.6, 0.65, 0.7, 0.8];
bounds = [-1, 3];

% Initialize results array - completely flat format
results = [];

% Test each dataset
for dataset_name = {'nile', 'sealevel'}
  name = dataset_name{1};
  data = eval(name);
  n = length(data);

  printf('\n%s (n=%d):\n', name, n);

  % Test each alpha value
  for i = 1:length(alpha_values)
    alpha = alpha_values(i);
    m = floor(n^alpha);

    printf('  m = %d: ', m);

    try
      % Optimize using veltaper with p=3 (Kolmogorov)
      obj_func = @(d) veltaper(d, data, m, 3);
      [d_hat, obj_val] = fminbnd(obj_func, bounds(1), bounds(2));

      % Standard error: sqrt(p * Phi / (4*m)) with p=3, Phi=1.00354
      se = sqrt(3 * 1.00354 / (4 * m));

      % Store test case info
      test_case = struct('dataset', name, 'm', m, 'd_hat', d_hat, 'se', se, 'obj', obj_val);
      results = [results; test_case];

      printf('d_hat = %.4f, se = %.4f\n', d_hat, se);

    catch
      printf('FAILED\n');
      % Store failed case with NaN values
      test_case = struct('dataset', name, 'm', m, 'd_hat', NaN, 'se', NaN, 'obj', NaN);
      results = [results; test_case];
    end
  end
end

% Export to JSON array of dictionaries
fid = fopen('octave_veltaper.json', 'w');
fprintf(fid, '[\n');
for i = 1:length(results)
  r = results(i);
  fprintf(fid, '  {"dataset": "%s", "m": %d, "d_hat": %.8f, "se": %.8f, "obj": %.8f}', ...
          r.dataset, r.m, r.d_hat, r.se, r.obj);
  if i < length(results)
    fprintf(fid, ',\n');
  else
    fprintf(fid, '\n');
  end
end
fprintf(fid, ']\n');
fclose(fid);

printf('\nResults saved to octave_veltaper.json\n');
