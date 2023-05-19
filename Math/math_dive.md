## Forward Pass

- set number of noise levels -> by Time step
- set noise level for each timestep
- Vecorize the noise levels

#### Steps

1. calculate the noise level $\beta$ for each timestep

   $$
     \beta = {x: start + (end - start) * x / (T-1), x = 0, 1, ..., T-1}
   $$

2. calculated the $\alpha$

$$
\alpha = 1 - \beta
$$

3. calculate the cumulative product of $\alpha$

$$
{\alpha_{cumprod}} = \prod_{t=1}^T \alpha_t
$$

4. calculate the cumulative product of alphas but the last noise level is set to 1 using padding

$$
\alpha_{previous-cumprod} = [1, {\alpha_{cumprod}}[0:T-1]]
$$

5. calculate square root of the reciprocal of $\alpha{cumprod}$

$$
\bar{\alpha} = \sqrt{\frac{1}{\alpha_{}}}
$$

6. calculate the square root of the cumulative product of $\alpha$

$$
\alpha_{sqrt-cumprod} = \sqrt{\alpha_{cumprod}} = \sqrt{\prod_{t=1}^T \alpha_t}
$$

7. calculated the square root of 1 minus the cumulative product of $\alpha$

$$
\alpha_{sqrt-one-minus-cumprod} = \sqrt{1 - \alpha_{cumprod}} = \sqrt{1 - \prod_{t=1}^T \alpha_t}
$$

8. Calculate the posterior varaince of the gaussian distribution
  
$$
\sigma_{posterior-variance} = \beta * \frac{1 - \alpha_{comprod-previous}}{1- \alpha_{cumprod}}
