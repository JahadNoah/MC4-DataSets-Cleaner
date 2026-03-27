"""
test_pipeline.py
================
Quick smoke-test: runs the pipeline on a tiny synthetic dataset
to verify keyword detection, 3R detection, and reporter output.

Run: python test_pipeline.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detectors.porn_detector import PornDetector
from detectors.sensitive_3r_detector import ThreeRDetector
from reporter import Reporter

# ---------------------------------------------------------------------------
# Synthetic test records
# ---------------------------------------------------------------------------
TEST_RECORDS = [
    # (expected_porn, expected_3r_categories, text)
    (False, [],            "Perdana Menteri melancarkan program baharu untuk pendidikan 2024."),
    (True,  [],            "Video lucah itu tersebar di internet, menampilkan aksi bogel."),
    (True,  [],            "The porn site had explicit nude videos and sexual content."),
    (False, ["RACE"],      "Cina celaka semua pergi balik China, jangan duduk Malaysia!"),
    (False, ["RELIGION"],  "Islam sesat, al-Quran semua dusta, nabi Muhammad tipu rakyat."),
    (False, ["ROYALTY"],   "Sultan zalim dan korup, patut kita hapus sistem beraja di Malaysia."),
    (True,  ["RACE"],      "Puki pundek keling babi, semua kaum India patut diusir!"),
    (False, ["RELIGION","ROYALTY"], "Agong kafir, gereja pun kena bakar, raja korup semua."),
    (False, [],            "Kajian menunjukkan peningkatan dalam sektor pelancongan negara."),
    (True,  [],            "Dia suka menonton xvideos dan pornhub setiap malam."),
]

# ---------------------------------------------------------------------------
# Autonomous verification test cases for 3R-Race
# These test that the verification layer correctly rescues false positives
# while still catching true hate speech.
# ---------------------------------------------------------------------------
VERIFICATION_TESTS = [
    # (should_be_flagged, description, text)

    # TRUE POSITIVES — should remain flagged after verification
    (True,  "explicit slur bigram",
     "Cina babi semua balik China, tak payah duduk sini!"),
    (True,  "targeted racial insult",
     "Orang melayu ni memang bodoh, tak boleh maju langsung."),
    (True,  "racial violence incitement",
     "Kill all chinese pendatang, deport them from our country!"),

    # FALSE POSITIVES — should be RESCUED by verification (not flagged)
    (False, "academic study about racism",
     "Kajian universiti mengenai isu racist di sekolah menengah menunjukkan penyelidikan data yang positif."),
    (False, "news report about racial harmony",
     "Menurut laporan Bernama, program perpaduan kaum di sekolah menunjukkan kesan positif bagi toleransi."),
    (False, "historical commemoration",
     "Peristiwa 13 Mei diperingati di muzium negara sebagai peringatan sejarah bersejarah untuk pendidikan."),
    (False, "parliament policy discussion",
     "Perlembagaan menjamin hak ketuanan melayu dalam konteks parlimen dan undang-undang negara."),
    (False, "positive diversity celebration",
     "Sambutan perayaan muhibbah meraikan kepelbagaian kaum dan perpaduan di Malaysia dengan penuh keharmonian."),
]


def run_tests():
    porn_det = PornDetector(use_ml=False)
    three_r_det = ThreeRDetector(use_ml=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = Reporter(output_dir=tmpdir, verbose=True)

        passed = 0
        failed = 0

        print("\n" + "="*70)
        print("MC4 PIPELINE SMOKE TEST")
        print("="*70 + "\n")

        for i, (exp_porn, exp_3r, text) in enumerate(TEST_RECORDS, start=1):
            porn_r = porn_det.detect(text)
            three_r = three_r_det.detect(text)
            reporter.report(line_no=i, text=text, dataset_id="test",
                            porn_result=porn_r, three_r_result=three_r)

            porn_ok = porn_r.is_explicit == exp_porn
            three_r_ok = set(exp_3r).issubset(set(three_r.categories))
            ok = porn_ok and three_r_ok

            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
                print(f"  [{status}] Record {i}")
                if not porn_ok:
                    print(f"         Porn: expected={exp_porn} got={porn_r.is_explicit} "
                          f"kw={porn_r.matched_keywords[:3]}")
                if not three_r_ok:
                    print(f"         3R:  expected={exp_3r} got={three_r.categories} "
                          f"race={three_r.race_matches[:2]} "
                          f"rel={three_r.religion_matches[:2]} "
                          f"roy={three_r.royalty_matches[:2]}")

        summary = reporter.finalize()

        print(f"\nTests: {passed} passed, {failed} failed out of {len(TEST_RECORDS)}")
        print(f"Reporter summary: {json.dumps(summary, indent=2)}\n")

        # Check output files exist
        report_dir = Path(tmpdir)
        for fname in ["flagged.jsonl", "flagged_summary.csv", "summary.json"]:
            fpath = report_dir / fname
            exists = fpath.exists()
            print(f"  Output file '{fname}': {'OK' if exists else 'MISSING'}")

        return failed == 0


def run_verification_tests():
    """Test the autonomous race-flag verification layer."""
    three_r_det = ThreeRDetector(use_ml=False)

    passed = 0
    failed = 0

    print("\n" + "="*70)
    print("3R-RACE AUTONOMOUS VERIFICATION TESTS")
    print("="*70 + "\n")

    for should_flag, desc, text in VERIFICATION_TESTS:
        result = three_r_det.detect(text)
        actual_flagged = result.race_flagged

        ok = actual_flagged == should_flag
        status = "PASS" if ok else "FAIL"

        if ok:
            passed += 1
        else:
            failed += 1

        verified_str = ""
        if result.race_verified is not None:
            verified_str = f" verified={result.race_verified} reason={result.verification_reason}"

        indicator = "✓" if ok else "✗"
        print(f"  [{status}] {indicator} {desc}")
        print(f"         expected_flag={should_flag} actual_flag={actual_flagged}{verified_str}")
        if result.race_matches:
            print(f"         matches={result.race_matches[:4]}")
        if not ok:
            print(f"         TEXT: {text[:120]}")
        print()

    print(f"Verification tests: {passed} passed, {failed} failed out of {len(VERIFICATION_TESTS)}")
    return failed == 0


if __name__ == "__main__":
    success1 = run_tests()
    success2 = run_verification_tests()
    sys.exit(0 if (success1 and success2) else 1)
