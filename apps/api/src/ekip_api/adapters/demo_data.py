from ekip_api.adapters.in_memory_retriever import StoredChunk

DEMO_CHUNKS: list[StoredChunk] = [
    StoredChunk(
        "tenant-dev",
        "dev-1",
        "boeing-mx-v1",
        "Fastener torque must be verified during the scheduled inspection of the landing gear.",
    ),
    StoredChunk(
        "tenant-dev",
        "dev-2",
        "boeing-mx-v2",
        "Revised procedure: torque verification now requires a calibrated wrench "
        "and a second inspector sign-off.",
    ),
    StoredChunk(
        "tenant-dev",
        "dev-3",
        "boeing-mx-v2",
        "Hydraulic line inspection interval reduced from 600 to 400 flight hours.",
    ),
    StoredChunk(
        "tenant-dev",
        "dev-4",
        "cabin-handbook",
        "Cabin lighting replacement procedure for the forward galley.",
    ),
    StoredChunk(
        "tenant-other",
        "oth-1",
        "other-doc",
        "Confidential torque inspection schedule belonging to another tenant.",
    ),
    StoredChunk("tenant-other", "oth-2", "other-doc", "Payroll processing calendar."),
]
