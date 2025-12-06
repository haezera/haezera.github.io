mpg=c(20.5,20.3,23.0,20.1,17.3,17.3,13.9,24.2,26.6,19.3,17.5,14.9,18.6,14.7,14.2,12.1,15.9,32.0,30.6,31.1,21.4,14.4,14.7,11.6,19.2,28.0,27.4,32.4,13.2,18.9,12.4,22.7)
disp=c(129.7,152.0,11.6,234.8,328.3,230.1,381.9,146.8,174.5,193.1,147.1,304.3,251.3,266.3,442.0,436.4,461.2,43.6,118.2,65.3,79.7,306.4,329.0,336.6,385.7,108.5,162.9,101.0,347.3,141.0,292.5,105.7)
log.mpg=c(3.02,3.01,3.14,3.00,2.85,2.85,2.63,3.19,3.28,2.96,2.86,2.70,2.92,2.69,2.65,2.49,2.77,3.47,3.42,3.44,3.06,2.67,2.69,2.45,2.95,3.33,3.31,3.48,2.58,2.94,2.52,3.12)
log.disp=c(4.87,5.02,2.45,5.46,5.79,5.44,5.95,4.99,5.16,5.26,4.99,5.72,5.53,5.58,6.09,6.08,6.13,3.78,4.77,4.18,4.38,5.72,5.80,5.82,5.96,4.69,5.09,4.62,5.85,4.95,5.68,4.66)
cars=data.frame(mpg,disp,log.mpg,log.disp)

plot_residuals <- function(model) {
  par(mfrow=c(2,2))
  plot(model)
  par(mfrow=c(1, 1))
}

# Question 1
# Part 1.
# a)
normal_model <- lm(mpg~disp)
log_model <- lm(log.mpg~log.disp)
plot_residuals(normal_model)
plot_residuals(log_model)

"
While both residuals vs fitted graphs show shape, the log model
has a strong and obvious shape. This means that the linearity
assumption is likely to be broken. 

Furthermore, the log model includes a strong influential
observation (observation 3). Referring to the normality assumption,
the log model performs better here - however, the linearity
assumption is so strongly violated that the non-log dataset
should likely be used. 
"

# Part 2.
# b)
summary(log_model)
# 50.24%

# c)
# H_0: log.disp = 0
# H_1: log.disp != 0

# F-statistic: 30.29
# Null-distribution: F(1, 30)
# p-value = 5.617 * 10^{-6}

"
With a 1% level of significance, the p-value shows we can
reject the null hypothesis - implying that the 
model with log.disp as a predictor is better than the
intercept only model. 
"

# d)
# This is just the slope
# -0.27539

# e)
confint(log_model, level=0.97)
"
(-0.3893795, -0.1613927)
"

# f)
predict(log_model, data.frame(log.disp=log(290)), interval="confidence", level=0.99)
"
(2.698533, 2.946202)
"

# Question 2
cars$dummy = c(rep(0, 24), 1, rep(0, 7))
dummy_model <- lm(log.mpg~log.disp+dummy, data=cars)

# b)
summary(dummy_model)
"
t value = 0.984
null distribution = t(29)
p-value = 0.333

p-value > 0.01, therefore, we cannot reject the null hypothesis
that the dummy has statistical significance to the model.
"

# c) No. The test is testing whether the 25-th observation has
# statistical significance to the model. It also does not 
# externally calculate the variance.

rstudent(log_model)[25]
# r_i = 0.9841012, under t(29), as we externally studentize
# the observation

1 - pt(0.9841012, 29)
# 0.1666027. Still not statistically significant. 

# Question 3
mpg=c(21.9,21.3,21.6,21.2,18.0,17.3,17.2,22.6,18.5,19.2,16.3,17.6,14.4,16.5,10.5,12.8,14.0,32.8,28.7,33.4,24.0,14.8,14.2,11.5,17.8,27.9,29.8,29.8,13.0,19.9,16.2,21.8)
wt=c(2.740,2.915,2.154,3.185,3.340,3.350,3.950,2.952,2.580,3.444,3.244,4.230,3.346,3.958,5.260,5.742,5.251,2.248,1.383,1.763,2.793,3.428,3.299,3.606,3.663,2.017,2.642,
     1.437,2.800,2.790,3.724,2.828)
disp=c(178.0,166.0,83.1,253.5,345.0,208.5,417.0,111.0,55.3,168.2,138.2,299.8,218.2,302.5,473.5,507.7,425.9,85.9,40.9,60.3,169.3,304.2,283.6,314.9,372.7,91.3,195.6,83.7,
       295.5,148.0,324.1,128.2)
cyl=c(6,6,4,6,8,6,8,4,4,6,6,8,8,8,8,8,8,4,4,4,4,8,8,8,8,4,4,4,8,6,8,4)
hp=c(110,110,93,110,175,105,245,62,95,123,123,180,180,180,205,215,230,66,52,65,97,150,150,245,175,66,91,113,264,175,335,109)
mpg=data.frame(mpg,wt,cyl,disp,hp)

q3_model <- lm(mpg~wt+cyl+disp+hp, data=mpg)
# a)
# F-test
# H_0: wt = cyl = disp = hp = 0
# H_1: not all betas are 0
# F-statistic = 25.21
# Distribution: F(4, 27)
# p-value: 8.912 * 10^{-9}
summary(q3_model)

"
With a 5% level of significance, the F-test shows enough evidence
to reject the null hypothesis; concluding that the model
with the predictors is better than the model only containing
the intercept term. 
"

# b)
# F-test
# H_0: cycl = disp = hp = 0
# H_1: not all are 0
anova(lm(mpg~wt, data=mpg), q3_model)

# F-statistic = 8.0194
# Distribution: F(3, 27)
# p-value = 0.0005578
"
The F-test concludes, given a significance level of 5%, that
there is enough evidence to reject the null hypothesis.
This indicates that not all of the betas of cycl, disp
and hp are 0, and thus contribute to the model.
"

# c)
# F-test
# H_0: displacement = 0
# H_1: displacement != 0
anova(lm(mpg~wt, data=mpg), lm(mpg~wt+disp, data=mpg))
# F-statistic = 0.7911
# Distribution: F(1, 29)
# p-value = 0.3811
"
With a 5% level of significance, and a p-value of 0.3811,
there is not enough evidence to reject the null hypothesis.
This means that the addition of displacement to the model
with weight, does not benefit the model with statistical
significance. 
"

# d)
predict(q3_model, data.frame(wt=3.8,cyl=6,disp=220,hp=160), interval="prediction", level=0.92)
"
(11.60024, 23.24071)
"
