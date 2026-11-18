"""Tests for Task 3. Run:  python -m unittest -v test_phishing"""
import json, os, subprocess, sys, tempfile, unittest
import pandas as pd
import phishing_features as pf
import phishing_detector as pd_mod

HERE = os.path.dirname(os.path.abspath(__file__))

class TestFeatures(unittest.TestCase):
    def f(self, text):
        return dict(zip(pf.FEATURE_NAMES, pf.email_features(text)))
    def test_url_counting_and_ip(self):
        r = self.f("go to http://192.168.1.5/login or https://example.com/a")
        self.assertEqual(r["n_urls"], 2); self.assertEqual(r["has_ip_url"], 1)
        self.assertEqual(r["has_plain_http_url"], 1)
    def test_shortener_tld_at(self):
        self.assertEqual(self.f("see http://bit.ly/abc")["has_url_shortener"], 1)
        self.assertEqual(self.f("see http://pay-secure.xyz/x")["has_suspicious_tld"], 1)
        self.assertEqual(self.f("see http://paypal.com@evil.com/x")["has_at_in_url"], 1)
    def test_clean_email_has_no_url_flags(self):
        r = self.f("Hi team, lunch at noon.")
        self.assertEqual(r["n_urls"], 0); self.assertEqual(r["has_ip_url"], 0)
    def test_keywords_and_shouting(self):
        r = self.f("URGENT! Verify your password immediately!!!")
        self.assertGreaterEqual(r["n_urgent_keywords"], 3); self.assertEqual(r["n_exclamations"], 4)
    def test_malformed_url_does_not_crash(self):
        pf.email_features("click http://[broken/url now")
        pf.email_features("")
    def test_transformer_shape(self):
        m = pf.HandcraftedFeatures().fit_transform(["a", "b http://x.com"])
        self.assertEqual(m.shape, (2, len(pf.FEATURE_NAMES)))

class TestLabels(unittest.TestCase):
    def test_label_mapping(self):
        m = pd_mod.map_label
        self.assertEqual(m("Phishing Email"), 1); self.assertEqual(m("Safe Email"), 0)
        self.assertEqual(m("spam"), 1); self.assertEqual(m("ham"), 0); self.assertEqual(m(1), 1)
        with self.assertRaises(ValueError): m("banana")

class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        subprocess.run([sys.executable, os.path.join(HERE, "make_demo_dataset.py")],
                       cwd=cls.tmp, check=True, capture_output=True)
        cls.csv = os.path.join(cls.tmp, "demo_emails_SYNTHETIC.csv")
        cls.out = os.path.join(cls.tmp, "res")
        cls.metrics = pd_mod.main(["train", "--data", cls.csv, "--outdir", cls.out])
    def test_outputs_created(self):
        for n in ["phishing_model.joblib", "metrics.json", "confusion_matrix.png",
                  "classification_report.txt"]:
            self.assertTrue(os.path.getsize(os.path.join(self.out, n)) > 0, n)
    def test_metrics_consistent(self):
        cm = self.metrics["confusion_matrix"]
        self.assertEqual(sum(cm.values()), self.metrics["test_size"])
        acc = (cm["TP"] + cm["TN"]) / self.metrics["test_size"]
        self.assertAlmostEqual(acc, self.metrics["accuracy"], places=3)
    def test_no_train_test_leakage_duplicates_removed(self):
        X, y, info = pd_mod.load_dataset(self.csv)
        self.assertEqual(len(X), len(set(X)))
    def test_predict_obvious_cases(self):
        class A: pass
        a = A(); a.outdir = self.out; a.file = None
        a.text = "URGENT your PayPal account is suspended verify your password http://45.12.99.3/login"
        self.assertEqual(pd_mod.predict(a)[0], "Phishing")
        a.text = "Hi Priya, please find attached the project timeline. Thanks."
        self.assertEqual(pd_mod.predict(a)[0], "Safe")
    def test_unknown_column_message(self):
        bad = os.path.join(self.tmp, "bad.csv"); pd.DataFrame({"a": [1], "b": [2]}).to_csv(bad, index=False)
        with self.assertRaises(SystemExit): pd_mod.load_dataset(bad)

if __name__ == "__main__":
    unittest.main()
