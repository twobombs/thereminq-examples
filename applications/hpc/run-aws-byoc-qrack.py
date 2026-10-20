# example for aws BYOC for qrack in aws braket
# work in progress, code can be changed without notice
# code from AWS example, reviewed & modified by gemini2

#!/usr/bin/env python
# coding: utf-8

# ```bash
# cat ./requirements.txt
# ```
# ```text
# amazon-braket-sdk
# amazon-braket-qrack-simulator
# ```

from braket.jobs import hybrid_job
from pathlib import Path

@hybrid_job(
    device="local:unitaryfund/qrack",
    dependencies=Path('./requirements.txt')
)
def run_qrack_job():
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator

    device = LocalSimulator(backend="qrack")

    bell = Circuit().h(0).cnot(0, 1)
    task = device.run(bell, shots=100)
    bitstring_counts = task.result().measurement_counts
    print(bitstring_counts)

    return bitstring_counts


job = run_qrack_job()
print(job)


print(f"Job state: {job.state()}")


result = job.result(poll_interval_seconds=10)
print(f"Job state: {job.state()}")
print(f"Job result: {result}")


@hybrid_job(
    device="local:unitaryfund/qrack",
    dependencies=Path('./requirements.txt'),
)
def run_qrack_job():
    from pyqrack import QrackSimulator, QrackCircuit
    
    # implement example
    
    return {}
    
job = run_qrack_job()
print(job)



print(f"Algorithm specification: {job.metadata()['algorithmSpecification']}\n")
print(f"Classical compute instance configuration: {job.metadata()['instanceConfig']}")


from braket.aws import AwsSession
from braket.jobs.config import InstanceConfig
from braket.jobs.image_uris import retrieve_image, Framework

@hybrid_job(
    device="local:unitaryfund/qrack",
    dependencies=["amazon-braket-qrack-cuda-simulator==0.4.0"],
    instance_config=InstanceConfig(instanceType="ml.p3.2xlarge", instanceCount=1),
    image_uri=retrieve_image(Framework.PL_PYTORCH, AwsSession().region)
)
def run_qrack_job():
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator

    device = LocalSimulator(backend="qrack")

    bell = Circuit().h(0).cnot(0, 1)
    task = device.run(bell, shots=100)
    bitstring_counts = task.result().measurement_counts
    print(bitstring_counts)
    
    return bitstring_counts
    

job = run_qrack_job()
print(job)


# ```bash
# python deploy-image-build-project.py
# ```

import boto3

ecr_client = boto3.client('ecr')

image_uri = None
for repository in ecr_client.describe_repositories(repositoryNames=['amazon-braket-byoc-qrack'])['repositories']:
    repository_uri = repository["repositoryUri"]
    image_uri = f"{repository_uri}:latest"
    break
    
print(f"URI of our custom image: {image_uri}")


from braket.jobs import hybrid_job
from braket.jobs.config import InstanceConfig # Import InstanceConfig

@hybrid_job(
    device="local:unitaryfund/qrack",
    image_uri=image_uri,
    instance_config=InstanceConfig(instanceType="ml.p3.2xlarge", instanceCount=1) # GPU instance
)
def run_qrack_job():
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator

    device = LocalSimulator(backend="qrack")

    bell = Circuit().h(0).cnot(0, 1)
    print(device.run(bell, shots=100).result().measurement_counts)

job = run_qrack_job()
print(job)


job.state()
