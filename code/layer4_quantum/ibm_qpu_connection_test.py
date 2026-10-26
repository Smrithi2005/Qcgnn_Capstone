from getpass import getpass

from qiskit_ibm_runtime import QiskitRuntimeService


def main():
    token = getpass("IBM Quantum API key: ").strip()
    crn = input("IBM Quantum instance CRN: ").strip()

    if not token or not crn:
        print("API key and instance CRN are required.")
        return

    try:
        service = QiskitRuntimeService(
            channel="ibm_cloud",
            token=token,
            instance=crn,
        )
        backends = service.backends()
    except Exception as exc:
        print(f"IBM Quantum authentication or connection failed: {exc}")
        return

    print("IBM Quantum connection succeeded. Available backends:")
    for backend in backends:
        status = backend.status()
        print(
            f"- {backend.name}: operational={status.operational}, "
            f"pending_jobs={status.pending_jobs}"
        )


if __name__ == "__main__":
    main()
