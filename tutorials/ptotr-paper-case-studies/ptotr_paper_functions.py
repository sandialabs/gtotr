from __future__ import annotations

import gc
import os
import pickle
import urllib.request
from pathlib import Path

import adrt
import nibabel as nib
import numpy as np
import pandas as pd
import pyttb as ttb
from joblib import Parallel, delayed
from scipy import sparse
from scipy.stats import poisson
from tqdm import tqdm
from zipfile import ZipFile

from gtotr import ptotr_cp

##########################################################################
# 01 - ICEWS tensor autoregression
##########################################################################

FILES = [
    "events.2004.20150313083407.tab",
    "events.2005.20150313083555.tab",
    "events.2006.20150313083752.tab",
    "events.2007.20150313083959.tab",
    "events.2008.20150313084156.tab",
    "events.2009.20150313084349.tab",
    "events.2010.20150313084533.tab",
    "events.2011.20150313084656.tab",
    "events.2012.20150313084811.tab",
    "events.2013.20150313084929.tab",
    "events.2014.20160121105408.tab",
]

COUNTRIES = [
    "North Korea",
    "Georgia",
    "Japan",
    "Ukraine",
    "South Korea",
    "Russian Federation",
    "United Kingdom",
    "Germany",
    "China",
    "Turkey",
    "Taiwan",
    "United States",
    "Australia",
    "Egypt",
    "Iran",
    "France",
    "India",
    "Afghanistan",
    "Lebanon",
    "Syria",
    "Iraq",
    "Israel",
    "Pakistan",
    "Occupied Palestinian Territory",
    "Sudan",
]

QUADS = ["V+", "V-", "M+", "M-"]


V_POS = {
    "01", "010", "011", "012", "013", "014", "015", "016", "017", "018", "019",
    "02", "020", "021", "0211", "0212", "0213", "0214", "022", "023", "0231",
    "0232", "0233", "0234", "024", "0241", "0242", "0243", "0244", "025",
    "0251", "0252", "0253", "0254", "0255", "0256", "026", "027", "028",
    "03", "030", "031", "0311", "0312", "0313", "0314", "032", "033", "0331",
    "0332", "0333", "0334", "034", "0341", "0342", "0343", "0344", "035",
    "0351", "0352", "0353", "0354", "0355", "0356", "036", "037", "038", "039",
    "04", "040", "041", "042", "043", "044", "045", "046", "05", "050", "051",
    "052", "053", "054", "055", "056", "057",
}

M_POS = {
    "06", "060", "061", "062", "063", "064", "07", "070", "071", "072", "073",
    "074", "075", "08", "080", "081", "0811", "0812", "0813", "0814", "082",
    "083", "0831", "0832", "0833", "0834", "084", "0841", "0842", "085",
    "086", "0861", "0862", "0863", "087", "0871", "0872", "0873", "0874",
}

V_NEG = {
    "09", "090", "091", "092", "093", "094", "10", "100", "101", "1011",
    "1012", "1013", "1014", "102", "103", "1031", "1032", "1033", "1034",
    "104", "1041", "1042", "1043", "1044", "105", "1051", "1052", "1053",
    "1054", "1055", "1056", "106", "107", "108", "11", "110", "111", "112",
    "1121", "1122", "1123", "1124", "1125", "113", "114", "115", "116", "12",
    "120", "121", "1211", "1212", "1213", "122", "1221", "1222", "1223",
    "1224", "123", "1231", "1232", "1233", "1234", "124", "1241", "1242",
    "1243", "1244", "1245", "1246", "125", "126", "127", "128", "129", "13",
    "130", "131", "1311", "1312", "1313", "132", "1321", "1322", "1323",
    "1324", "133", "134", "135", "136", "137", "138", "1381", "1382",
    "1383", "1384", "1385", "139",
}

M_NEG = {
    "14", "140", "141", "1411", "1412", "1413", "1414", "142", "1421", "1422",
    "1423", "1424", "143", "1431", "1432", "1433", "1434", "144", "1441",
    "1442", "1443", "1444", "145", "1451", "1452", "1453", "1454", "15",
    "150", "151", "152", "153", "154", "16", "160", "161", "162", "1621",
    "1622", "1623", "163", "164", "165", "166", "1661", "1662", "1663", "17",
    "170", "171", "1711", "1712", "172", "1721", "1722", "1723", "1724",
    "173", "174", "175", "18", "180", "181", "182", "1821", "1822", "1823",
    "183", "1831", "1832", "1833", "184", "185", "186", "19", "190", "191",
    "192", "193", "194", "195", "196", "20", "200", "201", "202", "203",
    "204", "2041", "2042",
}


def classify_quad(code: str) -> str | None:
    """Map a CAMEO code to one of the four event categories."""
    if code in V_POS:
        return "V+"
    if code in V_NEG:
        return "V-"
    if code in M_POS:
        return "M+"
    if code in M_NEG:
        return "M-"
    return None


def build_week_lookup(start="2004-01-01", end="2014-07-02"):
    """Construct weekly time indices for the analysis period."""
    dates = pd.date_range(start=start, end=end, freq="D")
    n_weeks = len(dates) // 7
    dates = dates[: n_weeks * 7]

    week_starts = dates[::7]

    date_to_week = pd.Series(
        np.repeat(np.arange(n_weeks), 7),
        index=dates.normalize(),
    )

    return week_starts, date_to_week


def read_events(path: Path) -> pd.DataFrame:
    """Read and preprocess an extracted ICEWS .tab file."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    return _clean_events(df, path.name)


def read_events_from_zip(path: Path) -> pd.DataFrame:
    """Read and preprocess the .tab file contained in an ICEWS ZIP archive."""
    with ZipFile(path) as z:
        members = [
            member
            for member in z.namelist()
            if member.lower().endswith(".tab")
        ]

        if len(members) != 1:
            raise ValueError(
                f"Expected exactly one .tab file in {path.name}; "
                f"found {len(members)}: {members}"
            )

        with z.open(members[0]) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                dtype=str,
                low_memory=False,
            )

    return _clean_events(df, path.name)

def _clean_events(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Select and clean the ICEWS variables needed to construct the tensor."""
    needed = [
        "Event Date",
        "Source Country",
        "Target Country",
        "CAMEO Code",
    ]

    missing = [column for column in needed if column not in df.columns]

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}"
        )

    df = df[needed].copy()

    df["Source Country"] = (
        df["Source Country"]
        .str.replace('"', "", regex=False)
        .str.strip()
    )

    df["Target Country"] = (
        df["Target Country"]
        .str.replace('"', "", regex=False)
        .str.strip()
    )

    df["CAMEO Code"] = df["CAMEO Code"].astype(str).str.strip()

    df["Event Date"] = (
        pd.to_datetime(df["Event Date"], errors="coerce")
        .dt.normalize()
    )

    df = df.dropna(subset=["Event Date"])

    return df


def _read_event_file(data_dir: Path, filename: str) -> pd.DataFrame:
    """Read an ICEWS file, whether supplied as .tab or .tab.zip."""
    tab_path = data_dir / filename
    zip_path = data_dir / f"{filename}.zip"

    if tab_path.exists():
        print(f"Reading {tab_path.name}")
        return read_events(tab_path)

    if zip_path.exists():
        print(f"Reading {zip_path.name}")
        return read_events_from_zip(zip_path)

    raise FileNotFoundError(
        f"Could not find {filename} or {filename}.zip in {data_dir}"
    )


def build_icews_tensor(data_dir: Path):
    """
    Construct the 25 x 25 x 4 x 548 ICEWS tensor.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory containing the raw ICEWS .tab and/or .tab.zip files.

    Returns
    -------
    yall : numpy.ndarray
        Four-way tensor containing weekly ICEWS event counts.
    metadata : dict
        Metadata describing the tensor dimensions and time index.
    """
    data_dir = Path(data_dir)

    week_starts, date_to_week = build_week_lookup()

    country_to_idx = {
        country: i
        for i, country in enumerate(COUNTRIES)
    }

    quad_to_idx = {
        quad: i
        for i, quad in enumerate(QUADS)
    }

    frames = [
        _read_event_file(data_dir, filename)
        for filename in FILES
    ]

    df = pd.concat(frames, ignore_index=True)

    # Retain events for which both the source and target
    # belong to the selected set of 25 countries.
    df = df[
        df["Source Country"].isin(COUNTRIES)
        & df["Target Country"].isin(COUNTRIES)
    ].copy()

    # Map detailed CAMEO codes to the four event categories.
    df["quad"] = df["CAMEO Code"].map(classify_quad)
    df = df.dropna(subset=["quad"])

    # Assign each event to a weekly time index.
    df["week"] = df["Event Date"].map(date_to_week)
    df = df.dropna(subset=["week"]).copy()
    df["week"] = df["week"].astype(int)

    # Convert categorical labels to integer tensor indices.
    df["src_idx"] = df["Source Country"].map(country_to_idx)
    df["tgt_idx"] = df["Target Country"].map(country_to_idx)
    df["quad_idx"] = df["quad"].map(quad_to_idx)

    # Count events for each tensor entry.
    counts = (
        df.groupby(
            ["src_idx", "tgt_idx", "quad_idx", "week"],
            sort=False,
        )
        .size()
        .reset_index(name="count")
    )

    # Initialize all combinations to zero and fill in observed counts.
    yall = np.zeros(
        (
            len(COUNTRIES),
            len(COUNTRIES),
            len(QUADS),
            len(week_starts),
        ),
        dtype=np.int32,
    )

    yall[
        counts["src_idx"].to_numpy(),
        counts["tgt_idx"].to_numpy(),
        counts["quad_idx"].to_numpy(),
        counts["week"].to_numpy(),
    ] = counts["count"].to_numpy()

    metadata = {
        "countries": COUNTRIES,
        "quads": QUADS,
        "week_starts": week_starts,
        "weekly_totals": yall.sum(axis=(0, 1, 2)),
    }

    return yall, metadata



def build_autoregression_data(
    Z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the ICEWS autoregression covariates and responses.

    Parameters
    ----------
    Z : np.ndarray
        ICEWS tensor of shape (25, 25, 4, T).

    Returns
    -------
    X : np.ndarray
        Covariate tensor of shape (25, 25, 4, 3, T - 5).
        The fourth mode contains:
            1. an all-ones tensor,
            2. Y(t-1),
            3. 1/4 * sum_{f=2}^5 Y(t-f).

    Y : np.ndarray
        Response tensor of shape (25, 25, 4, T - 5).
    """
    if Z.ndim != 4:
        raise ValueError(f"Expected a 4-D tensor, got shape {Z.shape}")

    if Z.shape[:3] != (25, 25, 4):
        raise ValueError(
            f"Expected first three dimensions to be (25, 25, 4), "
            f"got {Z.shape[:3]}"
        )

    T = Z.shape[3]

    if T < 6:
        raise ValueError("ICEWS data must contain at least 6 time points.")

    n = T - 5

    X = np.zeros((25, 25, 4, 3, n), dtype=float)

    # Mean/trend component
    X[:, :, :, 0, :] = 1.0

    # Lag-1 component: Y(t-1)
    X[:, :, :, 1, :] = Z[:, :, :, 4:T - 1]

    # Longer-term component:
    # 1/4 * [Y(t-2) + Y(t-3) + Y(t-4) + Y(t-5)]
    X[:, :, :, 2, :] = (
        Z[:, :, :, 0:T - 5]
        + Z[:, :, :, 1:T - 4]
        + Z[:, :, :, 2:T - 3]
        + Z[:, :, :, 3:T - 2]
    ) / 4.0

    # Response: Y(t)
    Y = Z[:, :, :, 5:T]

    return X, Y

    
def fit_ptotr_init(
    X: np.ndarray,
    Y: np.ndarray,
    rank: int,
    nit: int = 100,
    fit_tol: float = 1e-8,
    seed: int = 0,
) -> dict:
    """Fit one randomly initialized PToTR CP model."""

    model = ptotr_cp(
        responses=Y,
        covariates=X,
    )

    fit = model.fit(
        rank=rank,
        maxiters=nit,
        tolerance=fit_tol,
        epsDivZero=1e-10,
        seed=seed,
    )

    return {
        "fit": fit,
        "loglik": float(fit.llf),
    }
    

def fit_ptotr_rank(
    X: np.ndarray,
    Y: np.ndarray,
    rank: int,
    ninit: int = 100,
    nit: int = 100,
    fit_tol: float = 1e-8,
    n_jobs: int = 32,
    seed: int = 0,
) -> dict:
    """Fit a PToTR CP model using multiple random initializations."""

    if X.shape[-1] != Y.shape[-1]:
        raise ValueError(
            "X and Y must have the same number of observations: "
            f"{X.shape[-1]} != {Y.shape[-1]}"
        )
    rng = np.random.default_rng(seed)
    
    seeds = rng.integers(
        0,
        2**32 - 1,
        size=ninit,
    )
    
    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        return_as="generator",
    )(
        delayed(fit_ptotr_init)(
            X=X,
            Y=Y,
            rank=rank,
            nit=nit,
            fit_tol=fit_tol,
            seed=int(init_seed),
        )
        for init_seed in seeds
    )

    best_fit = None
    best_ll = -np.inf
    best_init = None

    for j, result in enumerate(results, start=1):
        print(
            f"rank={rank:2d} | "
            f"init={j:3d}/{ninit} | "
            f"loglik={result['loglik']:.6e}"
        )

        if result["loglik"] > best_ll:
            best_ll = result["loglik"]
            best_fit = result["fit"]
            best_init = j

    return {
        "fit": best_fit,
        "loglik": best_ll,
        "init_best": best_init,
        "rank": rank,
    }

def fit_ptotr_ranks(
    X,
    Y,
    ranks,
    ninit=100,
    nit=100,
    fit_tol=1e-8,
    n_jobs=32,
    output_dir="results",
):
    """Fit PToTR CP models over a collection of ranks."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ptotr_res.csv"

    # Number of observations
    n = Y.shape[-1]

    # Dimensions of predictor and response modes
    hs = X.shape[:-1]
    ms = Y.shape[:-1]

    results = []

    for rank in ranks:
        result = fit_ptotr_rank(
            X=X,
            Y=Y,
            rank=rank,
            ninit=ninit,
            nit=nit,
            fit_tol=fit_tol,
            n_jobs=n_jobs,
        )

        # Same parameter-count formula as the RMarkdown
        npar = rank * (
            sum(h - 1 for h in hs)
            + sum(m - 1 for m in ms)
            + 1
        ) 

        bic = -2 * result["loglik"] + np.log(n) * npar

        result["npar"] = npar
        result["bic"] = bic

        results.append(result)

        # Checkpoint after every rank
        results_df = pd.DataFrame([
            {
                "rank": r["rank"],
                "loglik": r["loglik"],
                "bic": r["bic"],
                "npar": r["npar"],
                "init_best": r["init_best"],
            }
            for r in results
        ])

        results_df.to_csv(output_file, index=False)

        print(
            f"\nFinished rank={rank:2d} | "
            f"loglik={result['loglik']:.6e} | "
            f"BIC={bic:.6e} | "
            f"npar={npar} | "
            f"best init={result['init_best']}\n"
        )

    return results

    
##########################################################################
# 02 - PET image reconstruction
##########################################################################

def ptotr_pet_download_data():

    ptotr_pet_data_urls = [
        'https://s3.amazonaws.com/openneuro.org/ds003011/sub-03/ses-Y1MRC/anat/sub-03_ses-Y1MRC_run-1_T1w.nii.gz',
        'https://s3.amazonaws.com/openneuro.org/ds003011/sub-03/ses-Y1MRC/anat/sub-03_ses-Y1MRC_run-2_T1w.nii.gz',
        'https://s3.amazonaws.com/openneuro.org/ds003011/sub-03/ses-Y1MRC/anat/sub-03_ses-Y1MRC_run-3_T1w.nii.gz',
        'https://s3.amazonaws.com/openneuro.org/ds003011/sub-03/ses-Y1MRC/anat/sub-03_ses-Y1MRC_run-4_T1w.nii.gz'
    ]

    for url in ptotr_pet_data_urls:
        local_filename = "../data/" + url.split('/')[-1]
        if not os.path.exists(local_filename):
            urllib.request.urlretrieve(url, local_filename)
            print(f"Downloading: {local_filename}")
        else:
            print(f"File already exists: {local_filename}")

def ptotr_pet_compute_B_true(savedata=None):

    FILENAME = f"../data/ptotr_pet_{savedata}.npy"

    if savedata:
        if os.path.exists(FILENAME):
            print(f"Loading B_true: {FILENAME}")
            B_true = np.load(FILENAME)
            return B_true

    print("Loading image data")
    Y1 = nib.load("../data/sub-03_ses-Y1MRC_run-1_T1w.nii.gz").get_fdata()
    Y2 = nib.load("../data/sub-03_ses-Y1MRC_run-2_T1w.nii.gz").get_fdata()
    Y3 = nib.load("../data/sub-03_ses-Y1MRC_run-3_T1w.nii.gz").get_fdata()
    Y4 = nib.load("../data/sub-03_ses-Y1MRC_run-4_T1w.nii.gz").get_fdata()

    print("Creating B_true")
    B_true = np.stack((
        (Y1 - Y1.min())/(Y1.max() - Y1.min()),
        (Y2 - Y2.min())/(Y2.max() - Y2.min()),
        (Y3 - Y3.min())/(Y3.max() - Y3.min()),
        (Y4 - Y4.min())/(Y4.max() - Y4.min()),
    ))

    B_true = 2*(B_true - B_true.min())/(B_true.max() - B_true.min())

    if savedata:
        FILENAME = f"../data/ptotr_pet_{savedata}.npy"
        print(f"Saving B_true: {FILENAME}")
        np.save(FILENAME, B_true)

    return B_true

def ptotr_pet_compute_Ys(B, savedata=None):

    FILEBASE = f"../data/ptotr_pet_{savedata}"
    FILENAME_YS = FILEBASE + '.npy'
    FILENAME_YS_IND = FILEBASE + '_ind.npy'

    if savedata:
        if os.path.exists(FILENAME_YS) and os.path.exists(FILENAME_YS_IND):
            print(f"Loading Ys: {FILENAME_YS}")
            Ys = np.load(FILENAME_YS)
            print(f"Loading Ys_ind: {FILENAME_YS_IND}")
            Ys_ind = np.load(FILENAME_YS_IND)
            return Ys, Ys_ind

    # generate indices for random 16% subset
    print("Creating Ys_ind (16%)")
    np.random.seed(1)
    Ys_ind = np.arange(262144)
    np.random.shuffle(Ys_ind)
    Ys_ind = Ys_ind[:(2*20976)]

    #applying radon transform
    print("Creating Ys")
    Zall = np.zeros((4,240,256,1024))+np.pi
    for aa in range(Zall.shape[0]):
        for bb in range(Zall.shape[1]):
            Zall[aa,bb,:,:] = adrt.utils.interp_to_cart(adrt.adrt(B[aa,bb,:,:]))

    # downscale
    Pall = Zall.reshape(4,240,256*1024)[:,:,Ys_ind]

    # generate Poisson rates and save
    Ys = np.random.poisson(Pall)

    if savedata:
        print(f"Saving Ys: {FILENAME_YS}")
        np.save(FILENAME_YS, Ys)
        print(f"Saving Ys indices: {FILENAME_YS_IND}")
        np.save(FILENAME_YS_IND, Ys_ind)

    return Ys, Ys_ind

def _process_chunk(p0, p1, arr, filename, shape, sz, dtype):
    """
    Compute one batch of impulse-image ADRTs and write into the memmap.

    Returns a small list of display transforms from this chunk.
    """
    # Re-open memmap inside each worker process
    Z = np.memmap(
        filename,
        dtype=dtype,
        mode="r+",
        shape=shape,
    )

    pixels = np.arange(p0, p1)
    aa, bb = np.divmod(pixels, sz)
    B = len(pixels)

    # Batch of one-hot images
    X = np.zeros((B, sz, sz), dtype=dtype)
    X[np.arange(B), aa, bb] = 1

    # Batched ADRT
    T = adrt.utils.interp_to_cart(adrt.adrt(X))

    # Flatten each transformed image and subsample using arr
    Z[p0:p1, :] = T.reshape(B, -1)[:, arr]

    # Save selected transforms for display
    disp_local = []
    for k in range(B):
        if aa[k] == bb[k] and aa[k] % 25 == 0:
            disp_local.append(T[k].copy())

    # Flush this worker's writes
    Z.flush()

    return disp_local

def ptotr_pet_compute_Xs(Y_ind, savedata=None):

    FILENAME = f"../data/ptotr_pet_{savedata}.tns"

    if savedata:
        if os.path.exists(FILENAME):
            print(f"Loading Xs: {FILENAME}")
            Xs_sparse = ttb.import_data(FILENAME)
            return Xs_sparse

    sz = 256
    m = len(Y_ind)
    dtype = np.float64

    batch_size = 64      # Tune based on memory
    n_jobs = -1          # Use all available cores; consider lowering if ADRT is multithreaded

    num_pixels = sz * sz

    memmap_filename = "../data/ptotr_pet_Xs_memmap_float64.dat"

    # Disk-backed output array.
    # Shape is flattened over pixels: pixel index p = aa * sz + bb.
    Xs = np.memmap(
        memmap_filename,
        dtype=dtype,
        mode="w+",
        shape=(num_pixels, m),
    )

    # Make sure arr is a NumPy array with integer dtype
    Y_ind = np.asarray(Y_ind, dtype=np.int64)

    # Define batch ranges
    chunks = [
        (p0, min(p0 + batch_size, num_pixels))
        for p0 in range(0, num_pixels, batch_size)
    ]

    # Run chunks in parallel, with tqdm progress as chunks complete
    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        return_as="generator",
    )(
        delayed(_process_chunk)(
            p0,
            p1,
            Y_ind,
            memmap_filename,
            Xs.shape,
            sz,
            dtype,
        )
        for p0, p1 in chunks
    )

    disp = []

    for disp_local in tqdm(results, total=len(chunks), desc="Creating Xs"):
        disp.extend(disp_local)

    # Final flush, remove memmap file
    Xs.flush()

    # reshape view using original indexing Zall[aa, bb, :]
    Xs = Xs.reshape(sz, sz, m)

    print("Converting to sparse tensor")
    nz = np.nonzero(Xs)
    subs = np.column_stack(nz)
    vals = Xs[nz]
    Xs_sparse = ttb.sptensor(subs, vals, Xs.shape)

    # clean up
    del Xs
    Path(memmap_filename).unlink()
    gc.collect()

    if savedata:
        print(f"Saving sparse Xs: {FILENAME}")
        ttb.export_data(Xs_sparse, FILENAME)

    return Xs_sparse

def pyttb_sptensor_to_X_csr(X_tensor, k=None, dtype=np.float64):
    """
    Convert a 3D pyttb.sptensor with shape (I, J, K) into a SciPy CSR matrix
    of shape (I * J, k), using pyttb/Fortran-order matricization.

    This matches the pyttb convention for:

        X_tensor.to_sptenmat(
            rdims=np.array([0, 1]),
            cdims=np.array([2])
        ).double()

    restricted to the first k columns.

    For tensor coordinate (i, j, t), the sparse matrix coordinate is:

        row = i + I * j
        col = t

    not:

        row = i * J + j

    because pyttb uses Fortran-order linearization.
    """
    if len(X_tensor.shape) != 3:
        raise ValueError(f"Expected a 3D sptensor. Got shape={X_tensor.shape}.")

    I, J, K = X_tensor.shape

    if k is None:
        k = K

    if k > K:
        raise ValueError(f"k={k} exceeds tensor third dimension K={K}.")

    subs = np.asarray(X_tensor.subs)
    vals = np.asarray(X_tensor.vals, dtype=dtype).reshape(-1)

    if subs.size == 0:
        return sparse.csr_matrix((I * J, k), dtype=dtype)

    # Keep only entries with third-mode index less than k.
    mask = subs[:, 2] < k

    i = subs[mask, 0].astype(np.int64, copy=False)
    j = subs[mask, 1].astype(np.int64, copy=False)
    t = subs[mask, 2].astype(np.int64, copy=False)

    # pyttb/Fortran-order row linearization for modes 0 and 1.
    rows = i + I * j
    cols = t
    data = vals[mask]

    X_csr = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(I * J, k),
        dtype=dtype,
    ).tocsr()

    X_csr.sum_duplicates()
    X_csr.eliminate_zeros()
    X_csr.sort_indices()

    return X_csr

def _llik_from_At(Yt, At, block_rows=4096):
    """
    Compute

        sum(Y * log(A) - A)

    using transposed arrays:

        Yt = Y.T
        At = A.T

    This avoids allocating a full extra log buffer.
    """
    total = 0.0
    n = At.shape[0]

    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)

        Ablk = At[start:stop]
        Yblk = Yt[start:stop]

        total += np.sum(Yblk * np.log(Ablk) - Ablk, dtype=np.float64)

    return float(total)

def ml_em_sparseX(
    Y,
    X_csr,
    stoptol=1e-10,
    maxiters=100,
    epsDivZero=1e-10,
    B=None,
    verb=False,
    callback=None,
    check_inputs=True,
):
    """
    Sparse-X Poisson vector-on-vector regression with identity link.

    Fits:

        Y ~ Poisson(B @ X)

    using multiplicative EM/MM updates, exploiting sparse X.

    Parameters
    ----------
    Y : ndarray, shape (m, n)
        Dense response matrix.

    X_csr : scipy.sparse matrix, shape (h, n)
        Sparse covariate matrix.

    stoptol : float
        Relative convergence tolerance based on loglikelihood.

    maxiters : int
        Maximum number of EM/MM iterations.

    epsDivZero : float
        Small positive constant used to avoid division by zero and log(0).

    B : ndarray, shape (m, h), optional
        Initial coefficient matrix. If None, initializes to ones.

    verb : bool
        Whether to print convergence messages.

    callback : callable, optional
        Called once per iteration as:

            callback(iteration, C, llik)

        where C = B.T has shape (h, m).

    check_inputs : bool
        If True, performs input validation.

    Returns
    -------
    B_est : ndarray, shape (m, h)
        Estimated coefficient matrix. This is returned as a C-contiguous copy.

    lliklast : float
        Final loglikelihood.

    lliks : ndarray
        Loglikelihood history.

    params : tuple
        Tuple containing (stoptol, maxiters, epsDivZero).

    Notes
    -----
    Internally this stores C = B.T, shape (h, m), so the two main sparse
    operations are:

        At = X.T @ C
        Nt = X @ (Y.T / At)

    where At = A.T.
    """
    if not sparse.issparse(X_csr):
        raise TypeError("X_csr must be a scipy sparse matrix.")

    X_csr = X_csr.tocsr().astype(np.float64, copy=False)
    X_csr.sum_duplicates()
    X_csr.eliminate_zeros()
    X_csr.sort_indices()

    Y = np.asarray(Y, dtype=np.float64, order="C")

    if check_inputs:
        if Y.ndim != 2:
            raise ValueError("Y must be a 2-dimensional array.")

        if np.any(Y < 0):
            raise ValueError("Y must be elementwise nonnegative.")

        if X_csr.data.size and np.any(X_csr.data < 0):
            raise ValueError("X must be elementwise nonnegative.")

    m, n = Y.shape
    h, nX = X_csr.shape

    if n != nX:
        raise ValueError(
            f"Y and X must have the same number of columns. "
            f"Got Y.shape={Y.shape} and X.shape={X_csr.shape}."
        )

    # Work with transposed dense response.
    # Shape: n x m.
    Yt = np.ascontiguousarray(Y.T)

    # Store C = B.T.
    # Shape: h x m.
    if B is None:
        C = np.ones((h, m), dtype=np.float64, order="C")
    else:
        B = np.asarray(B, dtype=np.float64)
        if B.shape != (m, h):
            raise ValueError(f"B must have shape {(m, h)}. Got {B.shape}.")
        C = np.ascontiguousarray(B.T)

    # Denominator for multiplicative update:
    #
    #     xsum[j] = sum_i X[j, i]
    #
    # Shape: h.
    xsum = np.asarray(X_csr.sum(axis=1)).ravel().astype(np.float64)
    np.maximum(xsum, epsDivZero, out=xsum)

    # Keep X.T as CSR so both multiplications are sparse @ dense.
    XT_csr = X_csr.T.tocsr()
    XT_csr.sum_duplicates()
    XT_csr.eliminate_zeros()
    XT_csr.sort_indices()

    # Initial fitted mean, transposed:
    #
    #     At = A.T = X.T @ B.T = X.T @ C
    #
    # Shape: n x m.
    At = XT_csr @ C
    np.maximum(At, epsDivZero, out=At)

    llik_prev = _llik_from_At(Yt, At)
    lliks = [llik_prev]
    lliklast = llik_prev
    convcrit = np.inf

    for rr in range(1, maxiters + 1):
        # Reuse At as:
        #
        #     Rt = Y.T / A.T
        #
        np.divide(Yt, At, out=At)

        # Numerator transpose:
        #
        #     Nt = X @ Rt
        #
        # Shape: h x m.
        Nt = X_csr @ At

        # Multiplicative update:
        #
        #     C = C * Nt / xsum[:, None]
        #
        C *= Nt
        C /= xsum[:, None]

        # New fitted mean, transposed.
        At = XT_csr @ C
        np.maximum(At, epsDivZero, out=At)

        lliklast = _llik_from_At(Yt, At)
        lliks.append(lliklast)

        denom = max(abs(lliklast), epsDivZero)
        convcrit = abs(lliklast - llik_prev) / denom

        if callback is not None:
            callback(rr, C, lliklast)

        if convcrit < stoptol:
            if verb:
                print(
                    "converged after",
                    rr,
                    "iterations with a loglikelihood of",
                    lliklast,
                )
            break

        llik_prev = lliklast

    else:
        if verb:
            print(
                "reached maximum iterations of",
                maxiters,
                "with a relative change in loglikelihood of",
                convcrit,
            )

    # Return B as a normal C-contiguous m x h array.
    B_est = np.ascontiguousarray(C.T)

    params = (stoptol, maxiters, epsDivZero)
    return B_est, lliklast, np.asarray(lliks), params

def rmse_from_C(C, B0, block_rows=2048):
    """
    Compute RMSE between C = B.T and B0 without forming the full temporary
    C - B0.T.

    Parameters
    ----------
    C : ndarray, shape (h, m)
        Transposed coefficient matrix.

    B0 : ndarray, shape (m, h)
        Reference coefficient matrix.

    block_rows : int
        Number of coefficient rows to process per block.

    Returns
    -------
    rmse : float
    """
    B0 = np.asarray(B0, dtype=np.float64)

    h, m = C.shape

    if B0.shape != (m, h):
        raise ValueError(f"B0 must have shape {(m, h)}. Got {B0.shape}.")

    ssq = 0.0

    for start in range(0, h, block_rows):
        stop = min(start + block_rows, h)

        diff = C[start:stop, :] - B0[:, start:stop].T
        ssq += np.sum(diff * diff, dtype=np.float64)

    return float(np.sqrt(ssq / B0.size))


def save_slices_from_C(
    C,
    saved_imgs,
    pp,
    iteration,
    image_shape=(256, 256),
    num_channels=4,
    num_slices=240,
    channel=0,
    sagittal_index=119,
    coronal_index=127,
    axial_index=127,
):
    """
    Save images for different views.
    
    """
    import numpy as np

    I, J = image_shape
    h, m = C.shape

    expected_h = I * J
    expected_m = num_channels * num_slices

    if h != expected_h:
        raise ValueError(f"C has h={h}, expected {expected_h}.")
    if m != expected_m:
        raise ValueError(f"C has m={m}, expected {expected_m}.")

    q0 = channel * num_slices
    z_indices = q0 + np.arange(num_slices)   # slices

    # axial
    i_indices = np.arange(I)
    axial_rows = i_indices + I * axial_index
    axial = C[axial_rows[:, None], z_indices[None, :]].T.copy()

    # coronal
    j_indices = np.arange(J)                 # image-plane columns = second index
    coronal_rows = coronal_index + I * j_indices
    coronal = C[coronal_rows[:, None], z_indices[None, :]].T.copy()

    # sagittal
    q_sagittal = q0 + sagittal_index
    sagittal = C[:, q_sagittal].reshape(I, J, order="F").copy()

    saved_imgs[(pp, iteration, "axial")]    = axial
    saved_imgs[(pp, iteration, "coronal")]  = coronal
    saved_imgs[(pp, iteration, "sagittal")] = sagittal

def fit_ml_em_sparse(
    Y,
    X_tensor,
    B0,
    pp_values=(1, 2, 4, 8, 16),
    p1=2622,
    itint=5,
    itmax=120,
    epsDivZero=1e-10,
    stoptol=-1,
    out_rmse_csv="../data/ml_em_rmses.csv",
    out_slices_pkl="../data/ml_em_slices.pkl",
    save_iterations=(10, 120),
    image_shape=(256, 256),
    num_channels=4,
    num_slices=240,
):
    """
    Perform sparse-X ML-EM fits for multiple data percentages.

    Each pp fit starts independently from an all-ones initial B.

    Parameters
    ----------
    Y : ndarray
        Dense response tensor. The code assumes the original usage:

            Y[:, :, :k].reshape(num_channels * num_slices, -1)

    X_tensor : pyttb.sptensor
        Sparse tensor with shape approximately:

            (256, 256, K)

        or more generally:

            (I, J, K)

    B0 : ndarray, shape (num_channels * num_slices, I * J)
        Reference coefficient matrix for RMSE.

    pp_values : tuple
        Values such as (1, 2, 4, 8, 16).

    p1 : int
        Number of third-mode samples corresponding to one percent.

    itint : int
        RMSE checkpoint interval.

    itmax : int
        Total EM/MM iterations per pp value.

    epsDivZero : float
        Small positive constant for numerical protection.

    stoptol : float
        Stopping tolerance. Use -1 to force all iterations.

    out_rmse_csv : str
        Output path for RMSE CSV.

    out_slices_pkl : str
        Output path for saved image slices.

    save_iterations : tuple
        Iterations at which image slices are saved.

    Returns
    -------
    rmse_df : pandas.DataFrame
        RMSE values.

    saved_imgs : dict
        Saved image slices.

    final_B_by_pp : dict
        Final B estimates by pp value.
    """
    I, J, K_total = X_tensor.shape

    if image_shape != (I, J):
        raise ValueError(
            f"image_shape={image_shape} does not match "
            f"X_tensor.shape[:2]={X_tensor.shape[:2]}."
        )

    m = num_channels * num_slices
    h = I * J

    B0 = np.asarray(B0, dtype=np.float64, order="C")

    if B0.shape != (m, h):
        raise ValueError(f"B0 must have shape {(m, h)}. Got {B0.shape}.")

    remses = []
    saved_imgs = {}
    final_B_by_pp = {}

    for pp in pp_values:
        k = p1 * pp

        if k > K_total:
            raise ValueError(
                f"For pp={pp}, k=p1*pp={k} exceeds "
                f"X_tensor third dimension {K_total}."
            )

        print(
            f"ml_em_sparseX, {pp}% data; iteration: 0",
            end=" ",
        )

        # Dense Y block.
        Ypp = Y[:, :, :k].reshape(m, -1)
        Ypp = np.asarray(Ypp, dtype=np.float64, order="C")

        # Sparse X block.
        X_csr = pyttb_sptensor_to_X_csr(X_tensor, k=k, dtype=np.float64)

        if Ypp.shape[1] != X_csr.shape[1]:
            raise ValueError(
                f"Ypp and X_csr column mismatch for pp={pp}: "
                f"Ypp.shape={Ypp.shape}, X_csr.shape={X_csr.shape}."
            )

        # Fresh independent initialization for each pp.
        Binit = np.ones((m, h), dtype=np.float64, order="C")

        remse0 = []

        def checkpoint(iteration, C, llik):
            if iteration % itint == 0:
                rmse = rmse_from_C(C, B0)
                remse0.append(rmse)
                print(iteration, end=" ", flush=True)

            if iteration in save_iterations:
                save_slices_from_C(
                    C=C,
                    saved_imgs=saved_imgs,
                    pp=pp,
                    iteration=iteration,
                    image_shape=image_shape,
                    num_channels=num_channels,
                    num_slices=num_slices,
                    channel=0,
                    sagittal_index=119,
                    coronal_index=127,
                    axial_index=127,
                )

        B, lliklast, lliks, params = ml_em_sparseX(
            Y=Ypp,
            X_csr=X_csr,
            maxiters=itmax,
            stoptol=stoptol,
            epsDivZero=epsDivZero,
            B=Binit,
            verb=False,
            callback=checkpoint,
            check_inputs=True,
        )

        print("")

        remses.append(remse0)
        final_B_by_pp[pp] = B

    iteration_labels = list(range(itint, itmax + 1, itint))
    percent_labels = [f"{pp}%" for pp in pp_values]

    rmse_df = pd.DataFrame(remses, index=percent_labels, columns=iteration_labels)
    rmse_df.to_csv(out_rmse_csv)
    print(f"Saved RMSEs to {out_rmse_csv}")

    with open(out_slices_pkl, "wb") as f:
        pickle.dump(saved_imgs, f)
    print(f"Saved image slices to {out_slices_pkl}")

    return rmse_df, saved_imgs, final_B_by_pp


def fit_ptotr_sparse(
    Y,
    X_tensor,
    B0,
    rank,
    pp_values=(1, 2, 4, 8, 16),
    p1=2622,
    itint=5,
    itmax=120,
    epsDivZero=1e-10,
    stoptol=-1,
    out_rmse_csv="../data/ptotr_rmses.csv",
    out_slices_pkl="../data/ptotr_slices.pkl",
    save_iterations=(10, 120),
    image_shape=(256, 256),
    num_channels=4,
    num_slices=240,
):
    """
    Perform sparse-X PToTR fits for multiple data percentages.

    Each pp fit starts independently from an all-ones initial B.

    Parameters
    ----------
    Y : pyttb.tensor
        Dense response tensor.

    X_tensor : pyttb.sptensor
        Sparse tensor with shape approximately:

            (256, 256, K)

        or more generally:

            (I, J, K)

    B0 : pyttb.tensor, shape (num_channels, num_slices, I, J)
        Reference coefficient tensor for RMSE.

    pp_values : tuple
        Values such as (1, 2, 4, 8, 16).

    p1 : int
        Number of third-mode samples corresponding to one percent.

    itint : int
        RMSE checkpoint interval.

    itmax : int
        Total iterations per pp value.

    rank: int
        Rank of the coefficient tensor to fit.

    epsDivZero : float
        Small positive constant for numerical protection.

    stoptol : float
        Stopping tolerance. Use -1 to force all iterations.

    out_rmse_csv : str
        Output path for RMSE CSV.

    out_slices_pkl : str
        Output path for saved image slices.

    save_iterations : tuple
        Iterations at which image slices are saved.

    Returns
    -------
    rmse_df : pandas.DataFrame
        RMSE values.

    saved_imgs : dict
        Saved image slices.

    final_B_by_pp : dict
        Final B estimates by pp value.
    """
    I, J, K_total = X_tensor.shape

    if image_shape != (I, J):
        raise ValueError(
            f"image_shape={image_shape} does not match "
            f"X_tensor.shape[:2]={X_tensor.shape[:2]}."
        )

    if B0.shape != (I, J, num_slices, num_channels):
        raise ValueError(f"B0 must have shape {(I, J, num_slices, num_channels)}. Got {B0.shape}.")

    channel        = 0    # saves slices from the first channel (same as ml_em)
    axial_slice    = 119
    coronal_index  = 127
    sagittal_index = 127

    remses = []
    saved_imgs = {}
    final_B_by_pp = {}

    # Prepare outputs
    percent_labels = [f"{pp}%" for pp in pp_values]
    iteration_labels = list(range(itint, itmax + 1, itint))
    remses = []       # list[ list[rmse] ] aligned to percent_labels x iteration_labels
    saved_imgs = {}   # optional
    final_B_by_pp = {}

    fit_tol = stoptol if (isinstance(stoptol, (int, float)) and stoptol > 0) else 1e-5

    for pp in pp_values:
        k = int(p1 * pp)
        if k > K_total:
            raise ValueError(
                f"For pp={pp}, k=p1*pp={k} exceeds X_tensor third dimension {K_total}."
            )

        print(f"ptotr_sparse, {pp}% data; iteration: 0", end=" ")

        Ypp = Y[:, :, :k]
        Xpp = X_tensor[:, :, :k]

        # Build Poisson/Identity model (sparse-aware per gtotr_cp.py)
        model = ptotr_cp(responses=Ypp, covariates=Xpp)

        # Initialize coeeficient tensor to all ones (when reconstructed using full())
        Bhat = (1/rank)*ttb.ktensor([np.ones((s,rank)) for s in B0.shape])

        # Iteration checkpoints by warm-starting in blocks of itint
        remse_pp = []
        for iter_end in iteration_labels:
            results = model.fit(
                init=Bhat,
                rank=rank,
                maxiters=itint,    # advance in blocks
                tolerance=fit_tol,
                epsDivZero=epsDivZero
            )
            Bhat = results.coef_

            rmse = np.sqrt(np.mean((B0-Bhat).data.reshape(-1,1)**2))
            remse_pp.append(rmse)
            print(iter_end, end=" ", flush=True)

            Bhat_arr = Bhat.to_tensor().data
            
            # ---- saved_imgs at snapshot ----
            if iter_end in save_iterations:
                saved_imgs[(pp, iter_end, "axial")]    = Bhat_arr[:, :, axial_slice, channel]
                saved_imgs[(pp, iter_end, "coronal")]  = Bhat_arr[:, coronal_index, :, channel]
                saved_imgs[(pp, iter_end, "sagittal")] = Bhat_arr[sagittal_index, :, :, channel]
                saved_imgs[(pp, iter_end, "loglike")]  = results.llf

        print("")  # newline after progress printout

        remses.append(remse_pp)
        final_B_by_pp[pp] = Bhat  # final normalized CP coef

    # Assemble RMSE DataFrame to match ml_em layout
    rmse_df = pd.DataFrame(remses, index=percent_labels, columns=iteration_labels)
    rmse_df.to_csv(out_rmse_csv)
    print(f"Saved RMSEs to {out_rmse_csv}")

    # Persist saved image slices (if collected)
    import pickle
    with open(out_slices_pkl, "wb") as f:
        pickle.dump(saved_imgs, f)
    print(f"Saved image slices to {out_slices_pkl}")

    return rmse_df, saved_imgs, final_B_by_pp


##########################################################################
# 03 - Changepoint dectection
##########################################################################

def ptotr_changepoint_gen_covariates(tau, len=14):
    pre_tau = np.tile(np.array([1,0]), (tau, 1)).T
    post_tau = np.tile(np.array([0,1]), (len-tau, 1)).T
    return ttb.tensor(np.hstack((pre_tau,post_tau)))

def ptotr_changepoint_create_data(num_students=10, num_topics=15, num_days=14, count_mean=1,
                change_topic=3, change_day=9, change_magnitude=20, random_seed=12345):

    samples = []
    np.random.seed(random_seed)
    for i in range(num_days):

        # Poisson rates
        rates = count_mean*np.ones((num_students, num_students, num_topics))
        if i >= change_day:
            rates[:,:,change_topic-1] *= change_magnitude

	# Poisson sampling
        data = poisson.rvs(rates)
        samples.append(data)

    # create sparse tensor
    subs_list = []
    vals_list = []
    for i in range(num_days):
        # data for the day
        data = samples[i]
        # sparse subscript extraction
        day_subs = np.array(data.nonzero())
        # sparse data extraction
        day_vals = data[tuple(day_subs)].reshape(-1,1)
        # add day to subscripts
        subs_list.append(np.vstack([day_subs,i*np.ones((1,day_subs.shape[1]))]).T)
        vals_list.append(day_vals)

    S = ttb.sptensor(
        subs=np.vstack(subs_list),
        vals=np.vstack(vals_list),
        shape=(num_students, num_students, num_topics, num_days)
    )

    return S
