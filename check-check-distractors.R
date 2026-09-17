library(tidyverse)
library(here)
theme_set(theme_bw())

file <- read_csv(here("output/bench_ns9_check_distilbert_claude.csv"))

foo <- file |>
  select(sentence, item_num, word_position, is_target_grammatical, is_distractor_grammatical) |>
  pivot_longer(`is_target_grammatical`:`is_distractor_grammatical`, names_to = "type", values_to = "value") |>
  mutate(value = as.numeric(value))

ggplot(foo, aes(x = word_position, y = value, color = type)) +
  stat_summary(fun.data = "mean_cl_boot") +
  coord_cartesian(xlim = c(0, 28))
file |> filter(!is_target_grammatical)

file |>
  filter(!is_target_grammatical) |>
  View()

file2 <- read_csv(here("output/bench_ns9_check_roberta_claude.csv"))

foo2 <- file2 |>
  select(sentence, item_num, word_position, is_target_grammatical, is_distractor_grammatical) |>
  pivot_longer(`is_target_grammatical`:`is_distractor_grammatical`, names_to = "type", values_to = "value") |>
  mutate(value = as.numeric(value))

ggplot(foo2, aes(x = word_position, y = value, color = type)) +
  stat_summary(fun.data = "mean_cl_boot") +
  coord_cartesian(xlim = c(0, 28))
file |> filter(!is_target_grammatical)

file2 |>
  filter(is_distractor_grammatical) |>
  View()
