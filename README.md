

---

# 0) One-time prerequisites (you can reuse later)

* Pick a Region (e.g., `us-east-1`) and an S3 bucket you own (must be globally unique), e.g., `sagemaker-<your-alias>-artifacts`.
* Make sure you have an IAM role that SageMaker can assume (if you run this from SageMaker Studio proper you can use `sagemaker.get_execution_role()`; from Studio **Lab** you’ll pass a role ARN string).

> Serverless Inference memory must be **1024–6144 MB**, and per-request payload is capped around **6 MB** with a **60s** processing time limit—keep your canvas PNG small. ([AWS Documentation][2], [SageMaker][3])

---

# 1) Add these cells to your **existing notebook** (after training)

> These assume your code followed the DataCamp example (`tf.keras.datasets.mnist.load_data()`, `/255.0`, 28×28, dense network). If you added layers or changed shapes, just keep the **28×28** input and **softmax(10)** output for compatibility. ([DataCamp][1])

### 1A) Export a TensorFlow **SavedModel**

```python
import os, tensorflow as tf

# Rebuild your trained model object if needed
# If you still have `model` in memory from training, reuse it directly.
# Ensure the model accepts (None, 28, 28) and outputs 10 classes.

# Add a batch/channel dimension wrapper if your model expects (28,28)
# We'll save a version that takes (None, 28, 28, 1) to be explicit.

# Wrap with a serving function that enforces correct shape
@tf.function(input_signature=[tf.TensorSpec([None, 28, 28, 1], tf.float32)])
def serve_fn(x):
    # If your model was built on (28,28) without channel dim, flatten/reshape accordingly:
    # Example for a simple Dense model created with Flatten(input_shape=(28,28)):
    x2 = tf.reshape(x, (-1, 28, 28))     # back to (N, 28, 28) for your Flatten layer
    return {"probs": model(x2)}           # name output key for TF Serving

export_dir = "exported_savedmodel/1"      # versioned directory
os.makedirs(export_dir, exist_ok=True)
tf.saved_model.save(model, export_dir, signatures={"serving_default": serve_fn})
print("SavedModel at", export_dir)
```

**Why SavedModel?** SageMaker’s **TensorFlow Serving** container expects a **SavedModel** inside your `model.tar.gz`. No custom `inference.py` is required if you use the TF Serving image. ([SageMaker][4])

### 1B) Package the model as `model.tar.gz` (correct structure)

```python
import tarfile, os

# The tarball must contain the SavedModel root (versioned folder with saved_model.pb, variables/)
# at the top level, not nested under an extra directory.
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("exported_savedmodel", arcname="")  # puts "1/" at the root

# Inspect
with tarfile.open("model.tar.gz", "r:gz") as tar:
    print("TAR CONTENTS:")
    for m in tar.getmembers():
        if "saved_model.pb" in m.name or m.name.endswith("variables/"):
            print(" ", m.name)
```

You should see paths like:

```
1/saved_model.pb
1/variables/variables.data-00000-of-00001
1/variables/variables.index
```

### 1C) Upload the artifact to **S3**

```python
import boto3, sagemaker
from datetime import datetime

session = boto3.session.Session()
region = session.region_name or "us-east-1"     # set explicitly if needed
s3 = boto3.client("s3", region_name=region)

bucket = "sagemaker-<your-alias>-artifacts"     # <-- change me
prefix = "mnist-tf"
key = f"{prefix}/artifacts/{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}/model.tar.gz"

s3.upload_file("model.tar.gz", bucket, key)
model_data = f"s3://{bucket}/{key}"
model_data
```

---

# 2) Deploy a **Serverless** TensorFlow endpoint

```python
!pip install -q "sagemaker>=2.200.0"
```

```python
from sagemaker.tensorflow import TensorFlowModel
from sagemaker.serverless import ServerlessInferenceConfig
import sagemaker, time

# If you're in SageMaker Studio proper, you can:
# role = sagemaker.get_execution_role()
# In Studio Lab, provide an execution role ARN with SageMaker hosting + S3 read:
role = "arn:aws:iam::<ACCOUNT_ID>:role/<SageMakerExecutionRole>"

sm_sess = sagemaker.Session()
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=1024,  # MNIST is tiny; start at 1024 and raise if needed
    max_concurrency=2
)

tf_model = TensorFlowModel(
    model_data=model_data,
    role=role,
    framework_version="2.12",    # or a supported version in your region
)  # Using built-in TF Serving container; no entry_point/source_dir needed.

endpoint_name = f"mnist-tf-sls-{int(time.time())}"

predictor = tf_model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name=endpoint_name
)

print("Endpoint:", predictor.endpoint_name)
```

* Memory must be between **1–6 GB**; increase if you see OOM or heavy cold-starts. ([AWS Documentation][2], [SageMaker][3])
* Keep request bodies well under **6 MB**; MNIST PNGs of 28×28 are tiny, so you’re fine. ([AWS Documentation][5])

**(Optional) test quickly from the notebook**

```python
import numpy as np, json, base64, io
from PIL import Image, ImageOps

# draw a "fake" 5 or use X_test[0]
arr = np.zeros((28,28), dtype=np.uint8)
arr[5:23, 12:16] = 255

# encode to grayscale PNG base64
buf = io.BytesIO()
Image.fromarray(arr).save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

# We'll send a JSON instance to TF Serving compatible with our SavedModel signature
# (we'll decode/normalize in Lambda; here we show direct invoke to the endpoint runtime)
runtime = boto3.client("sagemaker-runtime", region_name=region)
resp = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps({"instances":[[[[p/255.0] for p in row] for row in arr.tolist()]]})  # (1,28,28,1)
)
print(json.loads(resp["Body"].read().decode())["predictions"][0][:5], "...")
```

---

# 3) **ETL + Inference API** (Lambda) for your drawing app

We’ll do preprocessing in Lambda: **accept base64 PNG**, convert to **28×28 grayscale**, **divide by 255.0**, and invoke the endpoint. Keep the image small to stay well under the payload limit. ([AWS Documentation][5])

**Lambda handler (Python 3.11):**

```python
# lambda_handler.py
import os, json, base64, io
import numpy as np
from PIL import Image, ImageOps
import boto3

RUNTIME = boto3.client("sagemaker-runtime")
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]

def preprocess_b64_png(b64_uri):
    raw = base64.b64decode(b64_uri.split(",")[-1])
    img = Image.open(io.BytesIO(raw))
    # ensure white background for transparency, then convert to grayscale
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255,255,255,255))
        bg.paste(img, mask=img.split()[-1])
        img = bg.convert("L")
    else:
        img = ImageOps.grayscale(img)
    img = img.resize((28,28), Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0  # [0,1], shape (28,28)
    arr = arr.reshape(1,28,28,1)                     # TF Serving expects NHWC
    return arr.tolist()

def handler(event, context):
    try:
        body = event.get("body")
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode()
        obj = json.loads(body)
        b64 = obj["b64"]

        instance = preprocess_b64_png(b64)
        payload = json.dumps({"instances": instance})

        resp = RUNTIME.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=payload,
        )
        pred = json.loads(resp["Body"].read().decode())
        probs = pred["predictions"][0]               # list of 10 floats
        top = int(np.argmax(probs))
        out = {"top_class": top, "probs": {str(i): float(p) for i,p in enumerate(probs)}}

        return {"statusCode": 200,
                "headers": {"content-type":"application/json","access-control-allow-origin":"*"},
                "body": json.dumps(out)}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
```

**Zip & deploy via CLI (minimal, with Function URL):**

```bash
REGION=us-east-1
FUNC=mnist-etl-infer
ENDPOINT_NAME=<your-endpoint-name>

zip function.zip lambda_handler.py

cat > trust.json <<'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
aws iam create-role --role-name LambdaSageMakerInvokeRole --assume-role-policy-document file://trust.json

cat > policy.json <<'JSON'
{"Version":"2012-10-17","Statement":[
  {"Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],"Resource":"*"},
  {"Effect":"Allow","Action":["sagemaker:InvokeEndpoint"],"Resource":"*"}
]}
JSON
aws iam put-role-policy --role-name LambdaSageMakerInvokeRole --policy-name LambdaInvokeSageMaker --policy-document file://policy.json
ROLE_ARN=$(aws iam get-role --role-name LambdaSageMakerInvokeRole --query Role.Arn --output text)

aws lambda create-function \
  --function-name $FUNC \
  --runtime python3.11 \
  --role $ROLE_ARN \
  --handler lambda_handler.handler \
  --zip-file fileb://function.zip \
  --region $REGION \
  --environment "Variables={ENDPOINT_NAME=$ENDPOINT_NAME}"

aws lambda create-function-url-config \
  --function-name $FUNC \
  --auth-type NONE \
  --cors "AllowOrigins=['*'],AllowMethods=['POST'],AllowHeaders=['*']" \
  --region $REGION

aws lambda get-function-url-config --function-name $FUNC --region $REGION --query FunctionUrl --output text
```

---

# 4) **Browser drawing app** (drop in an `index.html` and open locally)

This sends your canvas image to the Lambda Function URL from above:

```html
<!doctype html>
<html>
<head><meta charset="utf-8"/><title>Draw MNIST</title>
<style>body{font-family:system-ui;margin:20px}#canvas{border:1px solid #ccc;touch-action:none}</style></head>
<body>
  <h1>Draw a digit</h1>
  <canvas id="canvas" width="256" height="256"></canvas>
  <div style="margin-top:10px">
    <button id="clear">Clear</button>
    <button id="predict">Predict</button>
    <span id="status"></span>
  </div>
  <pre id="out"></pre>
<script>
const LAMBDA_URL = "https://xxxxxxxx.lambda-url.us-east-1.on.aws/"; // paste your URL

const c = document.getElementById('canvas'), ctx = c.getContext('2d');
ctx.lineWidth=18; ctx.lineCap='round'; ctx.lineJoin='round'; ctx.strokeStyle='#000';
let drawing=false;
function pos(e){const r=c.getBoundingClientRect(), t=e.touches?e.touches[0]:e; return {x:t.clientX-r.left, y:t.clientY-r.top};}
c.addEventListener('mousedown',e=>{drawing=true;const p=pos(e);ctx.beginPath();ctx.moveTo(p.x,p.y);});
c.addEventListener('mousemove',e=>{if(!drawing)return;const p=pos(e);ctx.lineTo(p.x,p.y);ctx.stroke();});
window.addEventListener('mouseup',()=>drawing=false);
c.addEventListener('touchstart',e=>{e.preventDefault();drawing=true;const p=pos(e);ctx.beginPath();ctx.moveTo(p.x,p.y);});
c.addEventListener('touchmove',e=>{e.preventDefault();if(!drawing)return;const p=pos(e);ctx.lineTo(p.x,p.y);ctx.stroke();});
window.addEventListener('touchend',()=>drawing=false);
document.getElementById('clear').onclick=()=>ctx.clearRect(0,0,c.width,c.height);
document.getElementById('predict').onclick=async()=>{
  document.getElementById('status').textContent='Sending...';
  const b64=c.toDataURL('image/png'); // tiny payload <<6MB limit
  const res=await fetch(LAMBDA_URL,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({b64})});
  const j=await res.json();
  document.getElementById('status').textContent='Done';
  document.getElementById('out').textContent=JSON.stringify(j,null,2);
};
</script>
</body></html>
```

---

# 5) “Answer-key” checklist your cohort can follow

**A. Notebook → model artifact**

1. Train MNIST with Keras (28×28, `/255.0`, labels 0–9). ([DataCamp][1])
2. Export **SavedModel** with a `serving_default` signature that accepts `(None,28,28,1)` and returns `"probs"`.
3. Pack as `model.tar.gz` with **SavedModel at top level** (e.g., `1/saved_model.pb`, `1/variables/*`). ([SageMaker][4])
4. Upload to `s3://<bucket>/<prefix>/artifacts/.../model.tar.gz`.

**B. Serverless endpoint**
5\. Create **TensorFlowModel** from `model_data`.
6\. Deploy with **ServerlessInferenceConfig** (`memory_size_in_mb` 1024–6144, `max_concurrency` as needed). ([AWS Documentation][2], [SageMaker][3])
7\. Note the `endpoint_name`.

**C. Lambda ETL + API**
8\. Create a Lambda with `sagemaker:InvokeEndpoint` rights.
9\. In handler: base64-decode PNG → grayscale → resize to 28×28 → `/255.0` → send `{"instances":[(1,28,28,1)]}` to endpoint.
10\. Return `{top_class, probs}`. Keep request size well under **6 MB**. ([AWS Documentation][5])
11\. Create a **Function URL** (or API Gateway) with CORS `*`.

**D. Front-end**
12\. Use `<canvas>` → `toDataURL('image/png')` → POST to Lambda.
13\. Display predicted digit + probabilities.

**E. Ops**
14\. Check CloudWatch logs/metrics for cold-start **OverheadLatency** and errors; increase memory if needed. ([AWS Documentation][2])
15\. Delete endpoint when done.

---

# 6) Suggested team split (2–3 per team)

**Team A — Model & Packaging (Keras/TF)**

* Confirm data normalization (`/255.0`), shape (28×28), and label order (0–9). ([DataCamp][1])
* Export **SavedModel** + create `model.tar.gz`.
* Provide 2–3 “golden” PNGs and expected top-1 digits.

**Team B — Platform & Deployment (SageMaker)**

* Create/reuse S3 bucket and role.
* Deploy **Serverless** TF endpoint; tune memory and note cold-start behavior. ([AWS Documentation][2])
* Share an `aws sagemaker-runtime invoke-endpoint` example for QA.

**Team C — App & ETL (Web + Lambda)**

* Build the canvas page and wire the **Function URL**.
* Implement Lambda preprocessing identical to training (28×28 grayscale, `/255.0`).
* Add error handling and simple UI for results.

**Integration hour**

* Team A’s sample PNGs → Team B endpoint returns stable JSON → Team C hits Function URL and verifies end-to-end.

---

## Why this works with your exact notebook

* The notebook’s MNIST example explicitly uses **TensorFlow/Keras** and **normalizes by `/255.0`** on **28×28** inputs. It predicts **10 digits**, so your class list is `["0","1","2","3","4","5","6","7","8","9"]`. ([DataCamp][1])
* The **TensorFlow Serving** container natively serves a **SavedModel** packaged inside `model.tar.gz`, which is perfect for Serverless endpoints; you don’t need a custom `inference.py`. ([SageMaker][4])
* **Serverless Inference** gives you pay-per-request with memory choices (1–6 GB) and small request limits (keep your canvas PNG small). ([AWS Documentation][2], [SageMaker][3])

If you want, I can also provide a **single bash script** that zips/deploys Lambda and prints the Function URL, plus a **minimal IaC** (SAM/CDK) so teams can redeploy this in one command.

[1]: https://www.datacamp.com/tutorial/sagemaker-studio-lab "AWS SageMaker Studio Lab: A Practical Hands-On Guide | DataCamp"
[2]: https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html?utm_source=chatgpt.com "Deploy models with Amazon SageMaker Serverless ..."
[3]: https://sagemaker.readthedocs.io/en/stable/api/inference/serverless.html?utm_source=chatgpt.com "Serverless Inference — sagemaker 2.250.0 documentation"
[4]: https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/deploying_tensorflow_serving.html?utm_source=chatgpt.com "Deploying to TensorFlow Serving Endpoints"
[5]: https://docs.aws.amazon.com/marketplace/latest/userguide/ml-service-restrictions-and-limits.html?utm_source=chatgpt.com "Service restrictions and quotas for machine learning ..."
