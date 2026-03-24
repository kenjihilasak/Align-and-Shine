import os
import shutil
import glob

# Rutas
SENT_ALIGN_DIR = "SentAlign"
UTILITIES_FILE = os.path.join(SENT_ALIGN_DIR, "utilities.pyx")
BACKUP_FILE = os.path.join(SENT_ALIGN_DIR, "utilities.pyx.original_backup")

# Código de Utilities con Margin Score (Completo para esa función)
MARGIN_CODE = """
# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

# Mantenemos imports necesarios del original si existen, pero redefinimos la función clave
@cython.boundscheck(False)
@cython.wraparound(False)
def create_labse_score_matrix(anchor_source_list, anchor_target_list, src_emb_dict, trg_emb_dict):
    # --- FORCED MARGIN SCORE PATCH ---
    # Calculates Ratio Margin: 2*cos(x,y) / (avg_src + avg_trg)
    
    cdef int k = 4
    cdef int n_src = len(anchor_source_list)
    cdef int n_trg = len(anchor_target_list)
    
    # 1. Vectorization (Numpy)
    # Note: We use try-except to handle potential missing keys gracefully
    try:
        src_vecs = np.array([src_emb_dict[s.strip()] for s in anchor_source_list])
        trg_vecs = np.array([trg_emb_dict[t.strip()] for t in anchor_target_list])
    except KeyError:
        return np.zeros((n_src, n_trg), dtype=np.float64)

    # 2. Cosine Similarity (Dot Product)
    sim_matrix = np.matmul(src_vecs, trg_vecs.T)
    
    # Validation for small matrices
    if n_src < k or n_trg < k:
        return sim_matrix
        
    # 3. Margin Calculation
    # Sort and take top-k neighbors
    src_top_k = np.sort(sim_matrix, axis=1)[:, -k:]
    src_avg = np.mean(src_top_k, axis=1)
    
    trg_top_k = np.sort(sim_matrix, axis=0)[-k:, :]
    trg_avg = np.mean(trg_top_k, axis=0)
    
    # 4. Ratio Margin Formula
    denominator = src_avg[:, np.newaxis] + trg_avg[np.newaxis, :]
    denominator[denominator < 1e-6] = 1e-6 
    
    return (2 * sim_matrix) / denominator
"""

def apply_patch():
    print(f"[1] Aplicando parche permanente a {UTILITIES_FILE}...")
    
    # Backup si no existe
    if not os.path.exists(BACKUP_FILE):
        shutil.copy(UTILITIES_FILE, BACKUP_FILE)
        print("    -> Backup creado en utilities.pyx.original_backup")
    
    # Leer archivo original
    with open(UTILITIES_FILE, 'r') as f:
        original_lines = f.readlines()
    
    # Reescribir archivo: Mantener todo EXCEPTO la función create_labse_score_matrix
    new_lines = []
    skip = False
    patched = False
    
    # Asegurar imports al inicio
    new_lines.append("import numpy as np\n")
    new_lines.append("cimport numpy as np\n")
    
    for line in original_lines:
        if "def create_labse_score_matrix" in line:
            skip = True
            if not patched:
                # Inyectar nuestra versión NUEVA solo una vez
                # Quitamos la cabecera del string MARGIN_CODE para no duplicar imports si ya estan
                code_body = MARGIN_CODE.split("def create_labse_score_matrix", 1)[1]
                new_lines.append("def create_labse_score_matrix" + code_body)
                patched = True
        
        if skip:
            # Detectar fin de la función (siguiente def/cdef/class o fin de indentación en cython es difícil, 
            # asumimos siguiente def)
            if (line.strip().startswith("def ") or line.strip().startswith("cdef ") or line.strip().startswith("class ")) and "create_labse_score_matrix" not in line:
                skip = False
                new_lines.append(line)
        else:
            if "import numpy" not in line and "cimport numpy" not in line: # Evitar duplicados
                new_lines.append(line)

    with open(UTILITIES_FILE, 'w') as f:
        f.writelines(new_lines)
    print("    -> Archivo utilities.pyx modificado exitosamente.")

def nuke_cache():
    print("[2] ELIMINANDO TODA CACHÉ DE COMPILACIÓN...")
    
    # 1. Carpetas locales
    for p in glob.glob(f"{SENT_ALIGN_DIR}/**/__pycache__", recursive=True):
        shutil.rmtree(p, ignore_errors=True)
    
    # 2. Archivos compilados locales (.so, .c) - ESTO ES CLAVE
    for ext in ["*.so", "*.c", "*.cpp", "*.html"]:
        for f in glob.glob(f"{SENT_ALIGN_DIR}/{ext}"):
            os.remove(f)
            print(f"    -> Borrado: {f}")

    # 3. Caché global de usuario (.pyxbld)
    home = os.path.expanduser("~")
    pyxbld = os.path.join(home, ".pyxbld")
    if os.path.exists(pyxbld):
        shutil.rmtree(pyxbld)
        print(f"    -> Borrado caché global: {pyxbld}")

if __name__ == "__main__":
    apply_patch()
    nuke_cache()
    print("\n[LISTO] Ahora ejecuta SentAlign normalmente.")