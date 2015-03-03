#include <Common/Base/keycode.cxx>

// This excludes libraries that are not going to be linked
// from the project configuration, even if the keycodes are
// present
#undef HK_FEATURE_PRODUCT_AI
#undef HK_FEATURE_PRODUCT_ANIMATION
#undef HK_FEATURE_PRODUCT_CLOTH
#undef HK_FEATURE_PRODUCT_DESTRUCTION_2012
#undef HK_FEATURE_PRODUCT_DESTRUCTION
#undef HK_FEATURE_PRODUCT_BEHAVIOR
#undef HK_FEATURE_PRODUCT_MILSIM
#undef HK_FEATURE_PRODUCT_PHYSICS

#define HK_EXCLUDE_LIBRARY_hkpVehicle
#define HK_EXCLUDE_LIBRARY_hkCompat
#define HK_EXCLUDE_LIBRARY_hkSceneData
#define HK_EXCLUDE_LIBRARY_hkcdCollide

//
// Common
//
#define HK_EXCLUDE_FEATURE_SerializeDeprecatedPre700
#define HK_EXCLUDE_FEATURE_RegisterVersionPatches 
//#define HK_EXCLUDE_FEATURE_MemoryTracker

//
// Physics
//
#define HK_EXCLUDE_FEATURE_hkpHeightField
//#define HK_EXCLUDE_FEATURE_hkpSimulation
//#define HK_EXCLUDE_FEATURE_hkpContinuousSimulation
//#define HK_EXCLUDE_FEATURE_hkpMultiThreadedSimulation

#define HK_EXCLUDE_FEATURE_hkpAccurateInertiaTensorComputer

#define HK_EXCLUDE_FEATURE_hkpUtilities
#define HK_EXCLUDE_FEATURE_hkpVehicle
#define HK_EXCLUDE_FEATURE_hkpCompressedMeshShape
#define HK_EXCLUDE_FEATURE_hkpConvexPieceMeshShape
#define HK_EXCLUDE_FEATURE_hkpExtendedMeshShape
#define HK_EXCLUDE_FEATURE_hkpMeshShape
#define HK_EXCLUDE_FEATURE_hkpSimpleMeshShape
#define HK_EXCLUDE_FEATURE_hkpPoweredChainData
#define HK_EXCLUDE_FEATURE_hkMonitorStream

#include <Common/Base/Config/hkProductFeatures.cxx>

// Platform specific initialization
#include <Common/Base/System/Init/PlatformInit.cxx>