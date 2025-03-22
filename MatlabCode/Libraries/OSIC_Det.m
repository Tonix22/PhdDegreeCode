function x_est = OSIC_Det(H, y)
  % Step 1: QR decomposition of H
  [Q, R] = qr(H);
  
  % Step 2: Transform the received vector using the conjugate transpose of Q
  v_tilde = Q' * y;
  
  % Step 3: Initialize the estimated signal vector
  N = size(H, 2);
  x_est = zeros(N, 1);  % Initialize as a column vector
  
  % Step 4: Backward substitution for interference cancellation
  for i = N:-1:1
      sum_Rx = R(i, i+1:N) * x_est(i+1:N);
      x_est(i) = (v_tilde(i) - sum_Rx) / R(i, i);
  end
end
