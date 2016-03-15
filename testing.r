library(tidyr)
library(ggplot2)

setwd('~/Repositories/holtwint')
df <- read.table('sampledata.csv', sep = ',', header = FALSE)

df$month <- seq_along(df$V1)
names(df) <- c('passengers', 'month')
ts <- ts(as.vector(df$passengers), frequency = 12)

ggplot(data = df[1:48,]) + geom_line(aes(x = month, y = passengers)) + scale_x_discrete(breaks = seq(0, 50, 1))
initfit <- lm(passengers ~ month, data = df[1:48,])
summary(initfit)

fit <- HoltWinters(ts, seasonal = 'additive', start.periods = 4)
round(fit$alpha, 3); round(fit$beta, 3); round(fit$gamma, 3)
round(fit$coefficients, 3); round(fit$SSE/nrow(df), 3)

pr24 <- predict(fit, n.ahead = 24, prediction.interval = FALSE)
round(pr24)
