from uagents import Agent, Bureau, Context, Model
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw
from typing import List
import os, random, csv, json
import datetime

# ========== CONFIG ==========
ROUNDS = 2
CANDIDATES_PER_ROUND = 10
IMAGE_DIR = "molecule_images"
RESULTS_BASE = "results"

# Create paths
job_id = datetime.datetime.now().strftime("job_%Y%m%d_%H%M%S")
results_path = os.path.join(RESULTS_BASE, job_id)
os.makedirs(results_path, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ========== MESSAGE MODELS ==========

class RequestDrugDiscovery(Model):
    target: str
    sequence: str

class MoleculeCandidate(Model):
    smiles: str
    logp: float
    mw: float
    tpsa: float
    affinity_score: float
    image_path: str
    toxicity: str
    round_id: int

class FinalSelection(Model):
    summary: str
    top_smiles: List[str]

# ========== AGENTS ==========

drug_discovery = Agent(name="drug_discovery", seed="drug_discovery_seed", port=8000, endpoint=["http://localhost:8000/submit"])
sales_rep = Agent(name="sales_rep", seed="sales_rep_seed", port=8001, endpoint=["http://localhost:8001/submit"])

# ========== CORE LOGIC ==========

def generate_valid_smiles(n: int) -> List[str]:
    base = ["CCO", "CCC", "CN", "CCN", "CNC", "COC", "CCl", "CCBr", "c1ccccc1", "c1ccncc1", "c1ccc(cc1)N"]
    results = set()
    while len(results) < n:
        parent = random.choice(base)
        mutated = parent + random.choice(["", "C", "O", "N", "Cl", "Br"])
        if Chem.MolFromSmiles(mutated):
            results.add(mutated)
    return list(results)

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {
        "logp": round(Descriptors.MolLogP(mol), 2),
        "mw": round(Descriptors.MolWt(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2)
    }

def mock_affinity(smiles, protein_seq):
    seed = hash(smiles + protein_seq) % 1000000
    random.seed(seed)
    return round(random.uniform(10, 100), 2)

def classify_toxicity(logp, mw, tpsa):
    if tpsa > 140:
        return "High TPSA"
    if logp > 5 or mw > 500:
        return "High logP/MW"
    return "Low Risk"

def draw_image(smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    img_path = os.path.join(IMAGE_DIR, f"{name}.png")
    Draw.MolToFile(mol, img_path, size=(300, 300))
    return img_path

def export_results(candidates: List[MoleculeCandidate]):
    csv_path = os.path.join(results_path, "top_candidates.csv")
    json_path = os.path.join(results_path, "top_candidates.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SMILES", "logP", "MW", "TPSA", "Affinity", "Toxicity", "Round", "Image"])
        for c in candidates:
            writer.writerow([c.smiles, c.logp, c.mw, c.tpsa, c.affinity_score, c.toxicity, c.round_id, c.image_path])

    with open(json_path, "w") as f:
        json.dump([c.dict() for c in candidates], f, indent=4)

# ========== DRUG DISCOVERY LOGIC ==========

@drug_discovery.on_message(model=RequestDrugDiscovery)
async def handle_discovery(ctx: Context, sender: str, msg: RequestDrugDiscovery):
    ctx.logger.info(f"💊 Starting discovery for target: {msg.target} (Job: {job_id})")
    all_candidates: List[MoleculeCandidate] = []

    for round_id in range(1, ROUNDS + 1):
        ctx.logger.info(f"🧬 Round {round_id}: generating molecules...")
        smiles_list = generate_valid_smiles(CANDIDATES_PER_ROUND)
        for smiles in smiles_list:
            props = calculate_descriptors(smiles)
            if props["logp"] < 3 and props["mw"] < 500:
                affinity = mock_affinity(smiles, msg.sequence)
                toxicity = classify_toxicity(props["logp"], props["mw"], props["tpsa"])
                image_path = draw_image(smiles, f"{job_id}_r{round_id}_{smiles}")
                all_candidates.append(MoleculeCandidate(
                    smiles=smiles,
                    logp=props["logp"],
                    mw=props["mw"],
                    tpsa=props["tpsa"],
                    affinity_score=affinity,
                    image_path=image_path,
                    toxicity=toxicity,
                    round_id=round_id
                ))

    top_candidates = sorted(all_candidates, key=lambda x: x.affinity_score)[:5]
    export_results(top_candidates)

    ctx.logger.info(f"✅ Discovery complete. Top {len(top_candidates)} candidates exported.")
    for i, mol in enumerate(top_candidates, 1):
        ctx.logger.info(f"{i}. {mol.smiles} | Aff: {mol.affinity_score} | {mol.toxicity}")

    await ctx.send(sales_rep.address, FinalSelection(
        summary=f"Top candidates discovered for {msg.target} (Job: {job_id})",
        top_smiles=[c.smiles for c in top_candidates]
    ))

# ========== SALES REP AGENT ==========

@sales_rep.on_interval(period=15.0)
async def start_discovery(ctx: Context):
    await ctx.send(drug_discovery.address, RequestDrugDiscovery(
        target="EGFR",
        sequence="MENSDLGAVVLGRGAFGKVV..."
    ))

@sales_rep.on_message(model=FinalSelection)
async def display_results(ctx: Context, sender: str, msg: FinalSelection):
    ctx.logger.info(f"[SalesRep] {msg.summary}")
    for i, smi in enumerate(msg.top_smiles, 1):
        ctx.logger.info(f"[SalesRep] Candidate {i}: {smi}")
    ctx.logger.info("[SalesRep] End of cycle.\n")

# ========== RUN BUREAU ==========

bureau = Bureau()
bureau.add(drug_discovery)
bureau.add(sales_rep)

if __name__ == "__main__":
    bureau.run()
