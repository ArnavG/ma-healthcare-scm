# Importing Excel data
library(readxl)
full_synth_data_excel <- read_excel("Desktop/SAAS_Sp22/full_synth_data_excel.xls")
View(full_synth_data_excel)

# Importing stats libraries
library(tidyverse)
library(haven)
library(Synth)
library(devtools)
if(!require(SCtools)) devtools::install_github("bcastanho/SCtools")
library(SCtools)

# Converting Excel data to R dataframe
full_synth_data_excel <- as.data.frame(full_synth_data_excel)

# Synthetic Control on All Forms of Health Insurance (Private and Medicaid)
dataprep_out_all <- dataprep(foo = full_synth_data_excel,
predictors = c("health_stat", "median_hhincome",
"mean_age", 
"u_rate",
"prop_fem", 
"prop_hispan", 
"prop_married",
"prop_ctzn",
"prop_pov",
"prop_edu"), 
predictors.op = "mean",
time.predictors.prior = 2000:2006, 
special.predictors = list(list("prop_uninsured", c(2000, 2002, 2004, 2006), "mean")), 
dependent = "prop_uninsured", 
unit.variable = "STATEFIP", 
time.variable = "YEAR",
treatment.identifier = 25,
controls.identifier = c(1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56),
time.optimize.ssr = 2000:2006,
time.plot = 2000:2009)

synth_out_all <- synth(data.prep.obj = dataprep_out_all)

# Synthetic Control Plots for All Health Insurance Coverage
path.plot(synth_out_all, dataprep_out_all, tr.intake = 2006, Ylab = c("Uninsured Individuals per 100 000"), Main = "All Health Care Coverage 2000-2009", Legend = c("MA", "Synthetic MA"))
gaps.plot(synth_out_all, dataprep_out_all, tr.intake = 2006, Ylab = c("Difference in Uninsured Individuals per 100 000"))

# Placebo Plots for Hypothesis Testing (All Health Coverage)
placebos_all <- generate.placebos(dataprep_out_all, synth_out_all, Sigf.ipop = 3)
plot_placebos(placebos_all)
mspe.plot(placebos_all, discard.extreme = FALSE, plot.hist = TRUE)

# Synthetic Control for Private Health Coverage
dataprep_out_priv <- dataprep(foo = full_synth_data_excel,
predictors = c("health_stat", "median_hhincome", 
"mean_age", 
"u_rate",
"prop_fem", 
"prop_hispan", 
"prop_married", 
"prop_ctzn", 
"prop_pov", 
"prop_edu"), 
predictors.op = "mean", 
time.predictors.prior = 2000:2006, 
special.predictors = list(list("priv_coverage", c(2000, 2002, 2004, 2006), "mean")), 
dependent = "priv_coverage", 
unit.variable = "STATEFIP", 
time.variable = "YEAR", 
treatment.identifier = 25, 
controls.identifier = c(1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56),
time.optimize.ssr = 2000:2006,
time.plot = 2000:2009)

synth_out_priv <- synth(data.prep.obj = dataprep_out_priv)

# Synthetic Control Plots for Private Health Coverage
path.plot(synth_out_priv, dataprep_out_priv, tr.intake = 2006, Ylab = c("Private Insurance Coverage per 100 000"), Main = "Private Coverage 2000-2009", Legend = c("MA", "Synthetic MA"))
gaps.plot(synth_out_priv, dataprep_out_priv, tr.intake = 2006, Ylab = c("Difference in Private Insurance Coverage per 100 000"))

# Placebo Plots for Hypothesis Testing (Private Coverage)
placebos_priv <- generate.placebos(dataprep_out_priv, synth_out_priv, Sigf.ipop = 3)
plot_placebos(placebos_priv)
mspe.plot(placebos_priv, discard.extreme = FALSE, plot.hist = TRUE)

# Synthetic Control for Medicaid Coverage
dataprep_out_mcaid <- dataprep(foo = full_synth_data_excel, 
predictors = c("health_stat", "median_hhincome", 
"mean_age", 
"u_rate", 
"prop_fem", 
"prop_hispan", 
"prop_married", 
"prop_ctzn", 
"prop_pov", 
"prop_edu"), 
predictors.op = "mean", 
time.predictors.prior = 2000:2006, 
special.predictors = list(list("mcaid_coverage", c(2000, 2002, 2004, 2006), "mean")), 
dependent = "mcaid_coverage", 
unit.variable = "STATEFIP", 
time.variable = "YEAR", 
treatment.identifier = 25, 
controls.identifier = c(1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56), 
time.optimize.ssr = 2000:2006, 
time.plot = 2000:2009)

synth_out_mcaid <- synth(data.prep.obj = dataprep_out_mcaid)

# Synthetic Control Plots for Medicaid Coverage
path.plot(synth_out_mcaid, dataprep_out_mcaid, tr.intake = 2006, Ylab = c("Medicaid Coverage per 100 000"), Main = "Medicaid Coverage 2000-2009", Legend = c("MA", "Synthetic MA"))
gaps.plot(synth_out_mcaid, dataprep_out_mcaid, tr.intake = 2006, Ylab = c("Difference in Medicaid Coverage per 100 000"))

# Placebo Plots for Hypothesis Testing (Medicaid Coverage)
placebos_mcaid <- generate.placebos(dataprep_out_mcaid, synth_out_mcaid, Sigf.ipop = 3)
plot_placebos(placebos_mcaid)
mspe.plot(placebos_mcaid, discard.extreme = FALSE, plot.hist = TRUE)

# Synthetic Control for Crude Mortality Rate
dataprep_out_mortality <- dataprep(foo = full_synth_data_excel,
predictors = c("median_hhincome",
"mean_age", 
"u_rate", 
"prop_fem", 
"prop_hispan", 
"prop_married",
"prop_ctzn",
"prop_pov",
"prop_edu"), 
predictors.op = "mean",
time.predictors.prior = 2000:2006, 
special.predictors = list(list("crude_rate", c(2000, 2002, 2004, 2006), "mean")), 
dependent = "crude_rate", 
unit.variable = "STATEFIP", 
time.variable = "YEAR",
treatment.identifier = 25,
controls.identifier = c(1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56),
time.optimize.ssr = 2000:2006,
time.plot = 2000:2009)

synth_out_mortality <- synth(data.prep.obj = dataprep_out_mortality)

# Synthetic Control Plots for Crude Mortality Rate
path.plot(synth_out_mortality, dataprep_out_mortality, tr.intake = 2006, Ylab = c("Crude Mortality Rate per 100 000"), Main = "Mortality Rates 2000-2009", Legend = c("MA", "Synthetic MA"))
gaps.plot(synth_out_mortality, dataprep_out_mortality, tr.intake = 2006, Ylab = c("Crude Mortality Rate per 100 000"))

# Placebo Plots for Hypothesis Testing (Crude Mortality Rate)
placebos_mortality <- generate.placebos(dataprep_out_mortality, synth_out_mortality, Sigf.ipop = 3)
plot_placebos(placebos_mortality)
mspe.plot(placebos_mortality, discard.extreme = FALSE, plot.hist = TRUE)