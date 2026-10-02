from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

# ----------------------------- PARAMÈTRES -----------------------------
CSV = "dvf_2025_mutations_propres.csv"
OUT = Path("sortie")
OUT.mkdir(exist_ok=True)

PERIMETRE = ["Maison", "Appartement"]   # filtre du périmètre (PAS une variable du k-means)
PRIX_MIN = 15_000                # plancher sur la valeur de la vente (ventes symboliques/erreurs). AUCUN plafond.
WINSOR = None                    # None = aucun écrêtage (les cas extrêmes restent tels quels) ; ex. (0.01, 0.99) pour écrêter
MIN_TX = 20                      # nb minimal de transactions par commune (essaie 30 pour des médianes plus stables)
K_RANGE = range(2, 13)           # k testés : AUCUNE restriction
K_FINAL = None                   # None = choix objectif ; sinon un entier pour forcer
N_BOOT = 20                      # nb de ré-échantillonnages pour la stabilité
ARI_MIN = 0.70                   # stabilité minimale acceptable
MIN_SHARE = 0.02                 # un cluster doit contenir au moins 2 % des communes
USE_REGION = False               # True = ajoute la région (one-hot) dans le k-means
SEED = 42

# ----------------------------- RÉGIONS -----------------------------
REGIONS = {
    "Auvergne-Rhône-Alpes": ["01", "03", "07", "15", "26", "38", "42", "43", "63", "69", "73", "74"],
    "Bourgogne-Franche-Comté": ["21", "25", "39", "58", "70", "71", "89", "90"],
    "Bretagne": ["22", "29", "35", "56"],
    "Centre-Val de Loire": ["18", "28", "36", "37", "41", "45"],
    "Corse": ["2A", "2B", "20"],
    "Grand Est": ["08", "10", "51", "52", "54", "55", "57", "67", "68", "88"],
    "Hauts-de-France": ["02", "59", "60", "62", "80"],
    "Île-de-France": ["75", "77", "78", "91", "92", "93", "94", "95"],
    "Normandie": ["14", "27", "50", "61", "76"],
    "Nouvelle-Aquitaine": ["16", "17", "19", "23", "24", "33", "40", "47", "64", "79", "86", "87"],
    "Occitanie": ["09", "11", "12", "30", "31", "32", "34", "46", "48", "65", "66", "81", "82"],
    "Pays de la Loire": ["44", "49", "53", "72", "85"],
    "Provence-Alpes-Côte d'Azur": ["04", "05", "06", "13", "83", "84"],
    "Outre-mer": ["971", "972", "973", "974", "976"],
}
DEPT2REG = {d: r for r, ds in REGIONS.items() for d in ds}


# ----------------------------- DONNÉES -----------------------------
def charger_mutations(path=CSV):
    cols = ["id_mutation", "annee_mois", "code_dept", "code_insee", "commune",
            "type_bien", "valeur_fonciere", "surface_habitable",
            "prix_m2_fiable", "retenue_stats"]
    df = pd.read_csv(path, usecols=cols, dtype={"code_dept": str, "code_insee": str})
    df = df[df["retenue_stats"].astype(str).eq("True")]
    df = df[df["type_bien"].isin(PERIMETRE)]
    df = df[df["valeur_fonciere"] >= PRIX_MIN]
    df = df[df["prix_m2_fiable"].notna() & (df["prix_m2_fiable"] > 0)]
    df = df.copy()
    df["region"] = df["code_dept"].map(DEPT2REG).fillna("Autre")
    return df


def agreger_communes(df):
    c = (df.groupby("code_insee")
           .agg(commune=("commune", "first"),
                code_dept=("code_dept", "first"),
                region=("region", "first"),
                n_tx=("prix_m2_fiable", "size"),
                prix_m2_med=("prix_m2_fiable", "median"),
                p25=("prix_m2_fiable", lambda s: s.quantile(0.25)),
                p75=("prix_m2_fiable", lambda s: s.quantile(0.75)),
                ticket_med=("valeur_fonciere", "median"),
                surf_med=("surface_habitable", "median"))
           .reset_index())
    c = c[c["n_tx"] >= MIN_TX].copy()
    c["dispersion"] = (c["p75"] - c["p25"]) / c["prix_m2_med"]   # proxy de risque
    return c.dropna(subset=["prix_m2_med", "ticket_med", "surf_med", "dispersion"]).reset_index(drop=True)


def construire_X(c):
    """log + (winsorisation optionnelle) + standardisation. Renvoie (X_scalé, noms_colonnes).
    Le ticket n'est PAS une variable (redondant avec prix x surface)."""
    F = pd.DataFrame({
        "log_prix_m2": np.log(c["prix_m2_med"]),
        "log_n_tx": np.log(c["n_tx"]),
        "dispersion": c["dispersion"],
        "surf_med": c["surf_med"],
    })
    if WINSOR:
        F = F.clip(lower=F.quantile(WINSOR[0]), upper=F.quantile(WINSOR[1]), axis=1)
    if USE_REGION:
        F = pd.concat([F, pd.get_dummies(c["region"], prefix="reg").astype(float)], axis=1)
    return StandardScaler().fit_transform(F), list(F.columns)

# ----------------------------- CHOIX DE k -----------------------------
def stabilite(X, k, ref_labels):
    """Ré-entraîne k-means sur 80 % des communes (N_BOOT fois), prédit toutes les communes,
    compare au clustering de référence avec l'ARI (1 = identique, 0 = hasard)."""
    rng = np.random.RandomState(SEED)
    aris = []
    for b in range(N_BOOT):
        idx = rng.choice(len(X), int(0.8 * len(X)), replace=False)
        m = KMeans(n_clusters=k, n_init=5, random_state=b).fit(X[idx])
        aris.append(adjusted_rand_score(ref_labels, m.predict(X)))
    return float(np.mean(aris)), float(np.std(aris))

def evaluer_k(X):
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        ari, ari_std = stabilite(X, k, km.labels_)
        rows.append({
            "k": k,
            "inertie": km.inertia_,
            "silhouette": silhouette_score(X, km.labels_, sample_size=min(10000, len(X)), random_state=SEED),
            "davies_bouldin": davies_bouldin_score(X, km.labels_),
            "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
            "stabilite_ari": ari, "stabilite_std": ari_std,
            "part_min_cluster": np.bincount(km.labels_).min() / len(X),
        })
        r = rows[-1]
        print(f"k={k:2d}  sil={r['silhouette']:.3f}  DB={r['davies_bouldin']:.3f}  "
              f"CH={r['calinski_harabasz']:.0f}  ARI={ari:.3f}±{ari_std:.3f}  min={r['part_min_cluster']:.1%}")
    return pd.DataFrame(rows)


def choisir_k(m):
    mm = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-12)
    m["score_global"] = (mm(m["silhouette"]) + mm(-m["davies_bouldin"]) + mm(m["calinski_harabasz"]) + mm(m["stabilite_ari"])) / 4
    m["eligible"] = (m["stabilite_ari"] >= ARI_MIN) & (m["part_min_cluster"] >= MIN_SHARE)
    pool = m[m["eligible"]] if m["eligible"].any() else m
    k = int(pool.loc[pool["score_global"].idxmax(), "k"])
    print("\nMeilleur k par critère :",
          f"silhouette={int(m.loc[m.silhouette.idxmax(), 'k'])},",
          f"DB={int(m.loc[m.davies_bouldin.idxmin(), 'k'])},",
          f"CH={int(m.loc[m.calinski_harabasz.idxmax(), 'k'])},",
          f"stabilité={int(m.loc[m.stabilite_ari.idxmax(), 'k'])}")
    return k


def main():
    df = charger_mutations()
    print(f"{len(df):,} mutations retenues (périmètre résidentiel, prix/m² fiable)")
    c = agreger_communes(df)
    print(f"{len(c):,} communes avec ≥ {MIN_TX} transactions")

    X, cols = construire_X(c)
    print("Variables du k-means :", cols)
    metriques = evaluer_k(X)
    k = K_FINAL if K_FINAL else choisir_k(metriques)
    metriques["k_retenu"] = metriques["k"].eq(k)
    metriques.to_csv(OUT / "metriques_k.csv", index=False)
    print(f"\n>>> k retenu = {k}" + ("  (forcé)" if K_FINAL else "  (règle objective)"))

    km = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(X)
    c["cluster"] = km.labels_
    ordre = c.groupby("cluster")["prix_m2_med"].median().sort_values().index   # 0 = le moins cher
    c["cluster"] = c["cluster"].map({old: new for new, old in enumerate(ordre)})

    c.to_csv(OUT / "communes_clusters.csv", index=False)
    profil = c.groupby("cluster").agg(
        n_communes=("code_insee", "size"), tx_total=("n_tx", "sum"),
        prix_m2_med=("prix_m2_med", "median"), ticket_med=("ticket_med", "median"),
        surf_med=("surf_med", "median"), tx_par_commune=("n_tx", "median"),
        dispersion=("dispersion", "median")).round(2)
    print("\nProfil des clusters :\n", profil)
    profil.to_csv(OUT / "profil_clusters.csv")


if __name__ == "__main__":
    main()