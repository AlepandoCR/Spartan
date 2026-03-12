package org.spartan.internal.model;

/**
 * Memory layout constants for C-compatible hyperparameter config structs.
 * <p>
 * Mirrors the Standard Layout of C++ structs in ModelHyperparameterConfig.h
 * All offsets are in BYTES.
 * <p>
 * <b>WARNING:</b> These offsets MUST match the C++ compiler's layout exactly.
 */
public final class SpartanConfigLayout {

    private SpartanConfigLayout() {}

    // ==================== BaseHyperparameterConfig ====================
    // Total size: 64 bytes (with alignment padding)
    public static final long BASE_MODEL_TYPE_OFFSET       = 0;   // int32_t
    public static final long BASE_LEARNING_RATE_OFFSET    = 8;   // double (aligned)
    public static final long BASE_GAMMA_OFFSET            = 16;
    public static final long BASE_EPSILON_OFFSET          = 24;
    public static final long BASE_EPSILON_MIN_OFFSET      = 32;
    public static final long BASE_EPSILON_DECAY_OFFSET    = 40;
    public static final long BASE_STATE_SIZE_OFFSET       = 48;  // int32_t
    public static final long BASE_ACTION_SIZE_OFFSET      = 52;  // int32_t
    public static final long BASE_IS_TRAINING_OFFSET      = 56;  // bool (1 byte + 7 padding)
    public static final long BASE_CONFIG_SIZE             = 64;

    // ==================== NestedEncoderSlotDescriptor ====================
    // 16 bytes (no padding - all int32_t)
    public static final long SLOT_START_INDEX_OFFSET      = 0;
    public static final long SLOT_ELEMENT_COUNT_OFFSET    = 4;
    public static final long SLOT_LATENT_DIM_OFFSET       = 8;
    public static final long SLOT_HIDDEN_COUNT_OFFSET     = 12;
    public static final long SLOT_DESCRIPTOR_SIZE         = 16;

    public static final int MAX_NESTED_ENCODER_SLOTS = 16;

    // ==================== RecurrentSoftActorCriticHyperparameterConfig ====================
    // Starts after BaseConfig (offset 64)
    public static final long RSAC_HIDDEN_STATE_SIZE_OFFSET                = 64;
    public static final long RSAC_RECURRENT_LAYER_DEPTH_OFFSET            = 68;
    public static final long RSAC_ACTOR_HIDDEN_NEURON_COUNT_OFFSET        = 72;
    public static final long RSAC_ACTOR_HIDDEN_LAYER_COUNT_OFFSET         = 76;
    public static final long RSAC_CRITIC_HIDDEN_NEURON_COUNT_OFFSET       = 80;
    public static final long RSAC_CRITIC_HIDDEN_LAYER_COUNT_OFFSET        = 84;
    public static final long RSAC_TARGET_SMOOTHING_OFFSET                 = 88;  // double
    public static final long RSAC_ENTROPY_ALPHA_OFFSET                    = 96;
    public static final long RSAC_FIRST_CRITIC_LR_OFFSET                  = 104;
    public static final long RSAC_SECOND_CRITIC_LR_OFFSET                 = 112;
    public static final long RSAC_POLICY_LR_OFFSET                        = 120;
    public static final long RSAC_RECURRENT_INPUT_FEATURE_COUNT_OFFSET    = 128; // int32_t
    public static final long RSAC_NESTED_ENCODER_COUNT_OFFSET             = 132;
    public static final long RSAC_REMORSE_BUFFER_CAPACITY_OFFSET          = 136;
    // 4 bytes padding for double alignment
    public static final long RSAC_REMORSE_SIMILARITY_THRESHOLD_OFFSET     = 144; // double
    public static final long RSAC_ENCODER_SLOTS_OFFSET                    = 152;

    /** Total byte size of RSAC config: 152 + (16 slots * 16 bytes) = 408 bytes */
    public static final long RSAC_CONFIG_TOTAL_SIZE = RSAC_ENCODER_SLOTS_OFFSET
            + (MAX_NESTED_ENCODER_SLOTS * SLOT_DESCRIPTOR_SIZE);

    // ==================== DDQN Config (future) ====================
    public static final long DDQN_TARGET_SYNC_INTERVAL_OFFSET    = 64;
    public static final long DDQN_REPLAY_BUFFER_CAPACITY_OFFSET  = 68;
    public static final long DDQN_REPLAY_BATCH_SIZE_OFFSET       = 72;
    public static final long DDQN_HIDDEN_NEURON_COUNT_OFFSET     = 76;
    public static final long DDQN_HIDDEN_LAYER_COUNT_OFFSET      = 80;
    public static final long DDQN_CONFIG_TOTAL_SIZE              = 88;

    // ==================== AutoEncoder Config (future) ====================
    public static final long AE_LATENT_DIM_SIZE_OFFSET           = 64;
    public static final long AE_ENCODER_HIDDEN_NEURON_OFFSET     = 68;
    public static final long AE_ENCODER_LAYER_COUNT_OFFSET       = 72;
    public static final long AE_DECODER_LAYER_COUNT_OFFSET       = 76;
    public static final long AE_BOTTLENECK_REG_WEIGHT_OFFSET     = 80;
    public static final long AE_CONFIG_TOTAL_SIZE                = 88;
}
