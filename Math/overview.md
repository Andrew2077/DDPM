forward pass:

- addding noise to the image on each timestep
- we can sample each timestep image independently

> warning : Math Dive

$$
q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})
$$

Noise On Sample =

$$
 q(x_t|x_{t-1}) = \mathcal{N}(x_t ; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

contional gaussian distribution with a mean dependent on the previous timestep image and a specific variance

- $x_t$ is the image at timestep
- $x_{t-1}$ is the image at the previous timestep, less noisy
- $\sqrt{1-\beta*t}x*{t-1}$ 
  - is the mean of the gaussian distribution

  - mean of the distribution = previous image \* noise level (depends on the variance sheduler $\beta_t$)

- $\beta_t I$ is the variance of the distribution, which is set to a constant value $\beta_t$ times the identity matrix $I$
- $ \beta $ controls how fast we converge toward a mean of zero which crossponds to a standard gaussian distribution
- `Add the right Amount of noise to the image at each timestep`
- we can sample each timestep image independently, we don't need to sample the whole sequence at once

  - base on $x_0$ we can sample $x_t$ for any $t$
    $$
    q(x_t|x_0) = N(x_t; \sqrt{\bar{\alpha_t}}x_0, (1-\bar{\alpha_t})I)
    $$
    where $\alpha_t$ is $1 - {\beta_t}$ which is the total amount of image left after all the noise is added
    , $\bar{\alpha_t}$ is the cumulative sum of the reciprocal of $\alpha_t$ from $t=1$ to $t=T$
    $$
      x_t = \sqrt{\bar{\alpha_t}}x_0 + \sqrt{1-\alpha_t}
    $$

  $$

[more details](https://www.youtube.com/watch?v=HoKDTa5jHvg&ab_channel=Outlier)
