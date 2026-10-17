library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(tidytext)
library(ggplot2)
library(forcats)
library(tibble)
library(scales)
library(quanteda)
library(quanteda.textstats)
library(udpipe)

# Approach 1: TF-IDF: Lexical Distinctiveness

# Import the 23 text files
files <- list.files(
  path = "/Users/nada/Downloads/Te_t_Files_export",
  pattern = "\\.txt$",
  full.names = TRUE,
  ignore.case = TRUE
)

# Read each file as one complete text
texts_raw <- lapply(files, read_file)

# Use filenames as document names
names(texts_raw) <- tools::file_path_sans_ext(basename(files))

# Normalize the long s (ſ) to modern s
texts_normalized <- lapply(texts_raw, function(txt) {
  str_replace_all(txt, "ſ", "s")
})

# Preserve document names
names(texts_normalized) <- names(texts_raw)

# Combine the 23 texts into a corpus
corp <- corpus(unlist(texts_normalized))

# Tokenize and clean texts
toks <- tokens(
  corp,
  remove_punct = TRUE,
  remove_numbers = TRUE,
  remove_symbols = TRUE
)

# Lowercase
toks <- tokens_tolower(toks)

# Remove stop words
toks <- tokens_remove(toks, pattern = stopwords("en"))

# Create document-feature matrix
dfm_mat <- dfm(toks)

# Calculate TF-IDF
dfm_tfidf <- dfm_tfidf(dfm_mat)

# Inspect the TF-IDF matrix
dfm_tfidf

# Convert TF-IDF results to a data frame
tfidf_df <- convert(
  dfm_tfidf,
  to = "data.frame"
)

# Convert from wide format to tidy format
tfidf_tidy <- tfidf_df %>%
  pivot_longer(
    cols = -doc_id,
    names_to = "word",
    values_to = "tfidf"
  )

# Inspect the tidy table
tfidf_tidy

# Extract the top 15 terms for each document
top_tfidf_terms <- tfidf_tidy %>%
  group_by(doc_id) %>%
  slice_max(
    order_by = tfidf,
    n = 15,
    with_ties = FALSE
  ) %>%
  arrange(doc_id, desc(tfidf)) %>%
  ungroup()

# Display the results
top_tfidf_terms

# Approach 2: Pearson correlation and similarity heatmap

# Calculate pairwise Pearson correlations between documents
sim_r <- textstat_simil(
  dfm_mat,
  margin = "documents",
  method = "correlation"
)

# Convert the similarity object to a matrix
r_mat <- as.matrix(sim_r)

# Check the dimensions
dim(r_mat)

# Inspect the first five documents
r_mat[1:5, 1:5]

# Round correlations to three decimal places
r_mat <- round(r_mat, 3)

# Convert the correlation matrix to long format
heat_df <- as.data.frame(r_mat) %>%
  rownames_to_column("doc_i") %>%
  pivot_longer(
    cols = -doc_i,
    names_to = "doc_j",
    values_to = "r"
  )

# Create the correlation heatmap
ggplot(heat_df, aes(x = doc_j, y = doc_i, fill = r)) +
  geom_tile() +
  coord_fixed() +
  scale_fill_gradient2(
    low = "blue",
    mid = "white",
    high = "red",
    midpoint = 0
  ) +
  labs(
    title = "Pearson Correlation Between Documents",
    x = NULL,
    y = NULL,
    fill = "Correlation"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1,
      size = 7
    ),
    axis.text.y = element_text(size = 7),
    panel.grid = element_blank()
  )

# Approach 3: Select two texts for syntactic complexity

selected_texts <- texts_normalized[
  c("A06785", "A06789")
]

# Prepare a data frame with one row per text
texts_df <- tibble(
  document = names(selected_texts),
  text = unlist(selected_texts, use.names = FALSE)
)

# Check selected texts 
texts_df %>%
  select(document, text)

# Download the English UDPipe model
model_info <- udpipe_download_model(
  language = "english-ewt"
)

# Load the model
ud_model <- udpipe_load_model(
  model_info$file_model
)

# Annotate both texts using UDPipe
anno_df <- texts_df %>%
  mutate(
    anno = purrr::map2(
      text,
      document,
      ~ udpipe_annotate(
        ud_model,
        x = .x,
        doc_id = .y
      ) %>%
        as.data.frame()
    )
  ) %>%
  select(anno) %>%
  unnest(anno) %>%
  rename(document = doc_id) %>%
  select(
    document,
    paragraph_id,
    sentence_id,
    token_id,
    token,
    lemma,
    upos,
    feats,
    head_token_id,
    dep_rel
  )

# Inspect the annotation results
glimpse(anno_df)

# Count annotated tokens in each document
anno_df %>%
  count(document)

# Identify syntactic complexity features
syntax_df <- anno_df %>%
  mutate(
    # Identify words rather than punctuation
    is_word = upos != "PUNCT",
    
    # Identify finite verbs as a proxy for clauses
    is_clause = (upos %in% c("VERB", "AUX")) &
      str_detect(coalesce(feats, ""), "VerbForm=Fin"),
    
    # Identify dependent clauses
    is_dep_clause = dep_rel %in% c(
      "advcl",
      "ccomp",
      "xcomp",
      "acl",
      "acl:relcl"
    ),
    
    # Identify coordination relations
    is_coord = dep_rel %in% c("conj", "cc"),
    
    # Identify complex nominal relations
    is_complex_nominal = dep_rel %in% c(
      "amod",
      "nmod",
      "compound",
      "appos"
    )
  )

# Inspect the identified syntactic features
syntax_df %>%
  select(
    document,
    token,
    upos,
    is_clause,
    is_dep_clause,
    is_coord,
    is_complex_nominal
  ) %>%
  head(20)

# Aggregate syntactic features at the sentence level
sentence_df <- syntax_df %>%
  filter(is_word) %>%
  group_by(document, sentence_id) %>%
  summarise(
    words = n(),
    clauses = sum(is_clause),
    dep_clauses = sum(is_dep_clause),
    .groups = "drop"
  )

# Inspect the sentence-level results
sentence_df

# View the first 10 sentences
sentence_df %>%
  slice_head(n = 10)

# 1. Mean Length of Sentence (MLS)
mls_df <- sentence_df %>%
  group_by(document) %>%
  summarise(
    MLS = mean(words),
    .groups = "drop"
  )

# 2. Clauses per Sentence (C/S)
clausal_density_df <- sentence_df %>%
  group_by(document) %>%
  summarise(
    sentences = n(),
    clauses = sum(clauses),
    C_per_S = clauses / sentences,
    .groups = "drop"
  )

# 3. Dependent Clauses per Clause and Sentence
subordination_df <- sentence_df %>%
  group_by(document) %>%
  summarise(
    clauses = sum(clauses),
    dep_clauses = sum(dep_clauses),
    sentences = n(),
    DC_per_C = dep_clauses / pmax(clauses, 1),
    DC_per_S = dep_clauses / sentences,
    .groups = "drop"
  )

# 4. Coordination per Clause and Sentence
coordination_df <- syntax_df %>%
  group_by(document) %>%
  summarise(
    coord_relations = sum(is_coord),
    clauses = sum(is_clause),
    sentences = n_distinct(sentence_id),
    Coord_per_C = coord_relations / pmax(clauses, 1),
    Coord_per_S = coord_relations / sentences,
    .groups = "drop"
  )

# 5. Complex Nominals per Clause and Sentence
nominal_df <- syntax_df %>%
  group_by(document) %>%
  summarise(
    complex_nominals = sum(is_complex_nominal),
    clauses = sum(is_clause),
    sentences = n_distinct(sentence_id),
    CN_per_C = complex_nominals / pmax(clauses, 1),
    CN_per_S = complex_nominals / sentences,
    .groups = "drop"
  )

# Combine all syntactic complexity measures
all_measures <- mls_df %>%
  left_join(
    clausal_density_df %>% select(document, C_per_S),
    by = "document"
  ) %>%
  left_join(
    subordination_df %>% select(document, DC_per_C, DC_per_S),
    by = "document"
  ) %>%
  left_join(
    coordination_df %>% select(document, Coord_per_C, Coord_per_S),
    by = "document"
  ) %>%
  left_join(
    nominal_df %>% select(document, CN_per_C, CN_per_S),
    by = "document"
  )

# Display the final summary table
all_measures %>%
  knitr::kable(
    digits = 2,
    col.names = c(
      "Document", "MLS", "C/S", "DC/C", "DC/S",
      "Coord/C", "Coord/S", "CN/C", "CN/S"
    )
  )

# Reshape the measures for plotting
syntax_long <- all_measures %>%
  pivot_longer(
    cols = -document,
    names_to = "Measure",
    values_to = "Value"
  ) %>%
  mutate(
    Category = case_when(
      Measure == "MLS" ~ "Sentence Length",
      Measure == "C_per_S" ~ "Clausal Density",
      Measure %in% c("DC_per_C", "DC_per_S") ~ "Subordination",
      Measure %in% c("Coord_per_C", "Coord_per_S") ~ "Coordination",
      Measure %in% c("CN_per_C", "CN_per_S") ~ "Phrasal Complexity"
    )
  )

# Plot the syntactic complexity profiles
ggplot(syntax_long, aes(x = Measure, y = Value, fill = document)) +
  geom_col(position = "dodge", width = 0.7) +
  facet_wrap(~Category, scales = "free", ncol = 2) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Syntactic Complexity Profiles: A06785 and A06789",
    subtitle = "Comparing five dimensions of syntactic complexity",
    x = NULL,
    y = "Value",
    fill = "Document"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top"
  )

# Extract sentences identified by UDPipe
sentence_examples <- anno_df %>%
  group_by(document, sentence_id) %>%
  summarise(
    sentence = paste(token, collapse = " "),
    words = sum(upos != "PUNCT"),
    clauses = sum(
      (upos %in% c("VERB", "AUX")) &
        str_detect(coalesce(feats, ""), "VerbForm=Fin")
    ),
    .groups = "drop"
  )

# Show longer sentences with several detected clauses
sentence_examples %>%
  filter(words >= 25, clauses >= 3) %>%
  group_by(document) %>%
  slice_max(order_by = clauses, n = 5, with_ties = FALSE) %>%
  ungroup() %>%
  arrange(document, desc(clauses))

# Display the complete candidate sentences

# Extract the complete sentence from A06785
sentence_06785 <- sentence_examples %>%
  filter(document == "A06785", sentence_id == 139) %>%
  pull(sentence)

# Extract the complete sentence from A06789
sentence_06789 <- sentence_examples %>%
  filter(document == "A06789", sentence_id == 482) %>%
  pull(sentence)



