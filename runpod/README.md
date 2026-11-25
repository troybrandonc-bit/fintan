# Fintan on RunPod

A rented GPU's own disk is wiped when the pod stops, so keep Fintan on a **network volume**: everything it learns
(updates, notes, settings) and the model download stay there between pods.

1. Create a network volume (50 GB is enough for Fintan-9B) in the region you will use.
2. Deploy a pod with:
   - Image: `ghcr.io/troybrandonc-bit/fintan:latest`
   - GPU: 24 GB or more for Fintan-9B (RTX 3090, 4090, A5000, L4, A40); 16 GB or more for Fintan-4B
   - Network volume: the one from step 1, mounted at `/workspace`
   - Expose HTTP port `8900`
   - Environment variables: `FINTAN_API_KEY` = a long random string (required), `FINTAN_MODEL` =
     `fintanlabs/Fintan-9B` or `fintanlabs/Fintan-4B`
3. When it has started (the first start downloads the model), its address is
   `https://<pod id>-8900.proxy.runpod.net/v1`. Use it with your key from any OpenAI-compatible app:

```bash
curl https://<pod id>-8900.proxy.runpod.net/v1/chat/completions -H "Authorization: Bearer $FINTAN_API_KEY" \
  -H "Content-Type: application/json" -d '{"model": "fintan-9b", "messages": [{"role": "user", "content": "Who are you?"}]}'
```

Stop the pod when you are done; start a new one on the same volume and everything it learnt is still there. To keep a
copy outside RunPod: `python scripts/backup_hf.py save --repo yourname/my-fintan` (your own private Hugging Face repo).
