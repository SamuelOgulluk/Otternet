CC = gcc
# CFLAGS: Flags pour la compilation (quand on utilise -c)
CFLAGS = -Wall -Wextra -g -fsanitize=address,undefined -fno-omit-frame-pointer -Ih
# LDFLAGS: Flags pour l'édition de liens (linker)
LDFLAGS = -lm -fsanitize=address,undefined # -lm est pour la maths, les sanitizers doivent aussi être ici

SRC_DIR = src
OBJ_DIR = obj
BIN = otter

# --- Fichiers de base (dans src/) ---
CORE_FILES = main ottertensors ottertensors_utilities ottertensors_random \
             ottermath otternet otternet_optimizers OtterLayers otternet_utilities \
             OtterActivation OtterDisplay OtterCuda
# 
# !! NOTE : Ajoutez le nom du fichier .c qui contient 'OTC_init' à la liste CORE_FILES !!
# Exemple: CORE_FILES += otter_cuda
#

# --- Fichiers d'opérations (dans src/operations/) ---
OP_FILES = operations/Otter_Matrix_multiply \
           operations/Otter_reset \
           operations/Otter_axes_sums \
           operations/Otter_copy \
           operations/Otter_dot_divide \
           operations/Otter_dot_prod \
           operations/Otter_sum \
           operations/Otter_sub \
           operations/Otter_Tensor_square \
           operations/Otter_slice \
           operations/Otter_sqrt \
           operations/Otter_Transpose \
           operations/Otter_scalars

# Conversion des noms de fichiers en chemins .c et .o
CORE_SRC = $(addprefix $(SRC_DIR)/,$(addsuffix .c,$(CORE_FILES)))
OP_SRC = $(addprefix $(SRC_DIR)/,$(addsuffix .c,$(OP_FILES)))

CORE_OBJ = $(addprefix $(OBJ_DIR)/,$(addsuffix .o,$(CORE_FILES)))
OP_OBJ = $(addprefix $(OBJ_DIR)/,$(addsuffix .o,$(OP_FILES)))

# Liste complète de tous les objets
OBJ = $(CORE_OBJ) $(OP_OBJ)

all: $(BIN)

$(BIN): $(OBJ)
# Ligne du Linker : combine CFLAGS et LDFLAGS
	$(CC) $(OBJ) -o $(BIN) $(CFLAGS) $(LDFLAGS)

# Règle pour les fichiers de base (ex: src/main.c -> obj/main.o)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Règle pour les fichiers dans 'operations' (ex: src/operations/foo.c -> obj/operations/foo.o)
$(OBJ_DIR)/operations/%.o: $(SRC_DIR)/operations/%.c | $(OBJ_DIR)/operations
	$(CC) $(CFLAGS) -c $< -o $@

# Cibles pour créer les répertoires automatiquement
$(OBJ_DIR) $(OBJ_DIR)/operations:
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN)

.PHONY: all clean