mpg=c(15.7, 16.1, 14.3, 18.4, 30.8, 25.7, 30.7, 16.4, 19.2, 15.7, 24.4, 22.8, 19.9, 19.5, 14.2, 22.9, 23.3, 19.1, 18.8, 18.4, 
      17.1, 15.8, 8.3, 10.6, 13.8, 33.0, 32.3, 34.8, 21.6)
disp=c(341.1, 303.3, 331.1, 416.0, 54.7, 136.8, 95.6, 340.5, 103.3, 295.1, 85.0, 192.9, 358.4, 244.2, 409.2, 216.3, 112.1, 
      136.2, 162.7, 250.1, 286.5, 345.2, 490.4, 476.8, 472.7, 2.8, 110.5, 80.7, 88.9)
log.mpg=c(2.75, 2.78, 2.66, 2.91, 3.43, 3.25, 3.42, 2.80, 2.95, 2.75, 3.19, 3.13, 2.99, 2.97, 2.65, 3.13, 3.14, 2.95, 2.93,
          2.91, 2.84, 2.76, 2.12, 2.36, 2.62, 3.5, 3.47, 3.55, 3.07)
log.disp=c(5.83, 5.71, 5.80, 6.03, 4, 4.92, 4.56, 5.83, 4.64, 5.69, 4.44, 5.26, 5.88, 5.50, 6.01, 5.38, 4.72, 4.91, 5.09, 5.52, 5.66, 5.84,
           6.2, 6.17, 6.16, 1.03, 4.71, 4.39, 4.49)
carsdat=data.frame(mpg,disp,log.mpg,log.disp)

plot_residuals <- function(model) {
  par(mfrow=c(2, 2))
  plot(model)
  par(mfrow=c(1, 1))
}

# Question 1.
# PART 1.

# a)
raw_model <- lm(mpg~disp)
raw_model_summary <- summary(raw_model)
log_model <- lm(log.mpg~log.disp)
log_model_summary <- summary(log_model)

plot_residuals(raw_model)
plot_residuals(log_model)

"
The raw model is better from an R^2 perspective, as well as a model
assumptions perspective. The log model suffers from significant
shape in the Residuals vs Fitted plot which indicates a
violation of the linearity assumption. 

The log model also significantly deviates from the normal QQ line,
violating the normality assumption.
"

# PART 2.
# b)
log_model_summary$fstatistic

# H_0: B_{log.disp} = 0
# H_1: B_{log.disp} != 0

# F statistic 31.64008
# Null distribution: F(1, 27)
# p-value: 5.715 * 10^{-6}

"
With a 1% level of significance, we can reject the null hypothesis.
This means that the model with log.disp is better than the
intercept only model. 
"

# c)
log_model_summary$coefficients["log.disp", "Estimate"]
# -0.241509

# d)
confint(log_model, level=0.99)
# (-0.3604691, -0.12225489)

# e)
predict(log_model, data.frame(log.disp=log(240)), interval="confidence", level=0.99)
# (2.769225, 3.017715)

# PART 3.
carsdat$dummy <- c(rep(0, 5), 1, rep(0, 23))
part_three_model <- lm(log.mpg~log.disp+dummy, data=carsdat)
# a)
summary(part_three_model)

# b)
"
With a t statistic of 0.972, a t distribution of t(26) and 
a p-value of 0.34, we cannot reject the null hypothesis, as
0.34 > 0.05. 
"

# Question 2
mpg=c(18.0, 24.2, 22.5, 22.7, 22.1, 16.0, 13.7, 23.9, 24.1, 17.7, 
      15.5, 17.4, 15.1, 15.8, 13.1, 7.5, 15.8, 30.2, 30.0, 33.0, 18.8,
      15.7, 16.9, 9.0, 19.1, 25.7, 24.9, 34.0, 15.4, 22.3, 15.5, 23.0)
wt=c(2.220, 3.305, 2.284, 3.393, 3.896, 3.174, 3.486, 3.118, 3.324, 3.240, 3.128, 
     4.208, 3.436, 3.860, 5.614, 5.032, 5.489, 1.908, 1.567, 1.713, 2.105,
     3.540, 3.667, 3.266, 3.831, 1.721, 1.988, 1.997, 3.114, 3.110, 3.638, 2.996)
disp=c(100.0, 224.5, 102.6, 284.7, 428.4, 182.1, 347.4, 135.9, 166.9, 137.6, 120.8, 296.5,
       231.7, 287.8, 526.6, 401.2, 461.6, 34.9, 68.5, 52.8, 66.1, 321.0, 338.8, 
       263.9, 397.9, 46.9, 97.5, 167.7, 342.6, 196.0, 311.2, 153.4)
cyl=c(6,6,4,6,8,6,8,4,4,6,6,8,8,8,8,8,8,4,4,4,4,8,8,8,8,4,4,4,8,6,8,4)
hp=c(110, 110, 93, 110, 175, 105, 245, 62, 95, 123, 123, 180, 180, 180, 205, 215, 230,
     66, 52, 65, 97, 150, 150, 245, 175, 66, 91, 113, 264, 175, 335, 109)
mpg=data.frame(mpg, wt, cyl, disp, hp)

# Part 1.
library(leaps)
x=as.matrix(mpg[,2:5])
y=mpg$mpg
subsets <- regsubsets(x=x, y=y)
subset_summary <- summary(subsets)

# a)
# 1 predictor: cyl
# 2 predictors: wt + cyl
# 3 predictors: wt + cyl + disp

# b)
subset_summary$adjr2
# R^2: 0.5996692, 0.6091114, 0.6976197
PRESS <- function(model) {
  p_res <- residuals(model) / (1 - hatvalues(model))
  return(sum(p_res^2))
}
PRESS(lm(mpg~cyl))
PRESS(lm(mpg~wt+cyl))
PRESS(lm(mpg~wt+cyl+disp))

"
The best model is the 3 predictor model, with the highest
adjusted R^2; which indicates that the regression
component explains more of the total variation, and
the lowest PRESS, which indicates the best out of sample
prediction.
"

# c)
"
PRESS is important to consider the out-of-sample prediction.
Using residuals to consider model prediction is a poor
test as the model is fit for the very sample. PRESS residuals
allow for the testing of out-of-sample prediction, and thus
can show relative predictive performance. 
"

# Part 2.
full_model <- lm(mpg~wt+cyl+disp+hp)
full_model_summary <- summary(full_model)
full_model_summary$fstatistic

# d)

# Hypotheses:
# H_0 : wt = cyl = disp = hp = 0
# H_1 : Not all betas are 0

# F-statistic: 20.51267
# Null distribution: F(4, 27)
# p-value: 7.294 * 10^{-8}

"
With a p-value < 0.05, we can reject the null hypothesis,
and conclude that the full model is better than the
intercept only model.
"

# e)

e_test <- anova(lm(mpg~wt), full_model)

# Hypotheses:
# H_0: cyl=disp=hp=0
# H_1: Not all betas are 0

# F statistic: 9.6683
# Null distribution: F(3, 27)
# p-value = 0.0001668

"
With a 5% level of significance, we have enough evidence
to reject the null hypothesis. This means that the additional
predictors do have a statistically significant positive
influence on the model.
"

# f)
predict(full_model, data.frame(wt=3.6, cyl=6, disp=220, hp=160), interval="prediction", level=0.98)
# (9.123288, 26.53452)


