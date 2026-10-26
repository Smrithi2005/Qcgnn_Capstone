from getpass import getpass

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


BACKEND_NAME = "ibm_fez"
SHOTS = 128


def main():
    token = getpass("IBM Quantum API key: ")
    crn = getpass("IBM Quantum instance CRN: ")
    if not token or not crn:
        print("API key and instance CRN are required.")
        return

    try:
        service = QiskitRuntimeService(
            channel="ibm_cloud",
            token=token,
            instance=crn,
        )
        backend = service.backend(BACKEND_NAME)
        status = backend.status()
    except Exception as exc:
        print(f"IBM Quantum connection failed: {exc}")
        return

    if not status.operational:
        print(f"Backend {BACKEND_NAME} is not operational. No job submitted.")
        return

    circuit = QuantumCircuit(2)
    circuit.ry(0.5, 0)
    circuit.cx(0, 1)
    circuit.measure_all()

    try:
        pass_manager = generate_preset_pass_manager(
            backend=backend,
            optimization_level=1,
        )
        isa_circuit = pass_manager.run(circuit)
    except Exception as exc:
        print(f"Circuit transpilation failed: {exc}")
        return

    print(f"Backend: {backend.name}")
    print(f"Operation counts: {isa_circuit.count_ops()}")
    print(f"Requested shots: {SHOTS}")
    if input("Type SUBMIT to submit one real hardware job: ") != "SUBMIT":
        print("Submission cancelled.")
        return

    try:
        sampler = SamplerV2(mode=backend)
        job = sampler.run([isa_circuit], shots=SHOTS)
        print(f"Job ID: {job.job_id()}")
        result = job.result()
        counts = result[0].data.meas.get_counts()
        print(f"Measurement counts: {counts}")
    except Exception as exc:
        print(f"IBM Quantum job failed: {exc}")


if __name__ == "__main__":
    main()
