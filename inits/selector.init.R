source("selector.R")

# Supported action selection policies.
source("selectors/eg.R")
source("selectors/egd.R")
source("selectors/exp4.R")
source("selectors/exp3.R")
source("selectors/edts.R")
source("selectors/dts.R")
source("selectors/rr.R")
source("selectors/ts.R")
source("selectors/ucb1.R")
source("selectors/avg.R")


SELECTORS <- list()

SELECTORS$eg <- PolicyEG$new(eps=0.05)
SELECTORS$rr <- PolicyRR$new()
SELECTORS$ts <- PolicyTS$new()
SELECTORS$avg <- PolicyAVG$new()

SELECTORS$edts.5.10 <- PolicyEDTS$new(C=5, epsilon=0.1)
SELECTORS$dts.5  <- PolicyDTS$new(C=5)
SELECTORS$dts.10 <- PolicyDTS$new(C=10)
SELECTORS$dts.20 <- PolicyDTS$new(C=20)
SELECTORS$dts.50 <- PolicyDTS$new(C=50)

SELECTORS$ucb1 <- PolicyUCB1$new()

SELECTORS$exp4 <- PolicyExp4$new(0.5)
SELECTORS$exp3 <- PolicyExp3$new(0.1)
SELECTORS$eg.dyn <- PolicyEGdyn$new(eps=0.1)

source("selectors/bruno.R")
SELECTORS$bruno <- PolicyBruno$new(epsilon=0.2)

