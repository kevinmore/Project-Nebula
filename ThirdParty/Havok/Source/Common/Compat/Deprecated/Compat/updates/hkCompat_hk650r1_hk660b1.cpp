/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Compat/Deprecated/Compat/hkCompatUtil.h>
#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Compat/Deprecated/Version/hkVersionRegistry.h>
#include <Common/Serialize/Version/hkVersionUtil.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>
#include <Common/Base/hkBase.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>

namespace hkCompat_hk650r1_hk660b1
{

#define REMOVED(TYPE) { 0,0, hkVersionRegistry::VERSION_REMOVED, TYPE, HK_NULL }
#define BINARY_IDENTICAL(OLDSIG,NEWSIG,TYPE) { OLDSIG, NEWSIG, hkVersionRegistry::VERSION_MANUAL, TYPE, HK_NULL }

	struct DummyArray
	{
		void* data;
		int size;
		int capacity;
	};

	static void Update_ignore( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
	}

	static void Update_assert( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		HK_ASSERT2(0xad904271, false, "This object is not expected to be updated directly.");
	}

	// allocate the memory for an array
	static void initArray( hkClassMemberAccessor& member, int numElements, hkObjectUpdateTracker& tracker )
	{
		DummyArray& dummyArray = *static_cast<DummyArray*>(member.getAddress());
		dummyArray.size = numElements;
		dummyArray.capacity = hkArray<char>::DONT_DEALLOCATE_FLAG | numElements;

		if( numElements > 0 )
		{
			int numBytes = numElements * member.getClassMember().getArrayMemberSize();
			dummyArray.data = hkAllocateChunk<char>( numBytes, HK_MEMORY_CLASS_SERIALIZE );
			tracker.addChunk( dummyArray.data, numBytes, HK_MEMORY_CLASS_SERIALIZE );
		}
	}

	static void Update_CheckBalanceModifier( hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker )
	{
		hkClassAccessor oldObj (oldVar);
		hkClassAccessor newObj (newVar);

		newObj.member("ragdollLeftFootBoneIndex").asInt16() = (hkInt16)oldObj.member("ragdollLeftFootBoneIndex").asInt32();
		newObj.member("ragdollRightFootBoneIndex").asInt16() = (hkInt16)oldObj.member("ragdollRightFootBoneIndex").asInt32();
	}

	static void Update_BalanceModifier( hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker )
	{
		hkClassAccessor oldObj (oldVar);
		hkClassAccessor newObj (newVar);

		newObj.member("ragdollLeftFootBoneIndex").asInt16() = (hkInt16)oldObj.member("ragdollLeftFootBoneIndex").asInt32();
		newObj.member("ragdollRightFootBoneIndex").asInt16() = (hkInt16)oldObj.member("ragdollRightFootBoneIndex").asInt32();

		hkClassMemberAccessor oldStepInfos(oldVar, "stepInfo");
		hkClassMemberAccessor newStepInfos(newVar, "stepInfo");

		DummyArray& oldStepInfoArray = *static_cast<DummyArray*>(oldStepInfos.getAddress());
		DummyArray& newStepInfoArray = *static_cast<DummyArray*>(newStepInfos.getAddress());

		int numStepInfo = oldStepInfoArray.size;

		int oldStepInfoSize = oldStepInfos.getClassMember().getStructClass().getObjectSize();
		int newStepInfoSize = newStepInfos.getClassMember().getStructClass().getObjectSize();

		char* oldStepInfoData = static_cast<char*>( oldStepInfoArray.data );
		char* newStepInfoData = static_cast<char*>( newStepInfoArray.data );

		// loop through all of the legs and copy the old event ID to the ID of the new event
		for( int i = 0; i < numStepInfo; i++ )
		{
			hkClassAccessor oldStepInfo( oldStepInfoData + i * oldStepInfoSize, &oldStepInfos.getClassMember().getStructClass() );
			hkClassAccessor newStepInfo( newStepInfoData + i * newStepInfoSize, &newStepInfos.getClassMember().getStructClass() );

			newStepInfo.member("boneIndex").asInt16() = (hkInt16)oldStepInfo.member( "boneIndex" ).asInt32();
		}
	}

	static void Update_BalanceModifierStepInfo( hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker )
	{
		HK_ASSERT2( 0x8a4f2e5c, 0, "This should never be called" );
	}

	static void Update_hkdDeformableBreakableShape( hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker )
	{
		hkClassAccessor oldObj (oldVar);

		// HKD-276 now stores the actual shape keys and does not assume childIdx == shapeKey anymore
		hkClassMemberAccessor newShapeKeyArray ( newVar, "childKeys");

		const int size = oldObj.member("origChildTransforms").asSimpleArray().size;
		// allocate memory for the array
		initArray( newShapeKeyArray, size, tracker);

		// as the former versions required a matching childIdx/shapeKey, we can assume this asset did work ok :)
		for (int i=0; i<size; ++i)
		{
			reinterpret_cast<hkUint32*> (newShapeKeyArray.asSimpleArray().data)[i] = i;
		}
	}

	static void Update_hkpEntity( hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker )
	{
		hkClassAccessor entity( newVar );
		hkClassAccessor motion( entity.member("motion").object() );
		hkUint8 type = motion.member("type").asUint8();
		if (type > 2)
		{
			if (type > 4)
			{
				type--;
			}

			type--;

			motion.member("type").asUint8() = type;
		}
	}

	static void Update_hkpMotion( hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker )
	{
		hkClassAccessor motion( newVar );
		hkUint8 type = motion.member("type").asUint8();
		if (type > 2)
		{
			if (type > 4)
			{
				type--;
			}

			type--;

			motion.member("type").asUint8() = type;
		}
	}

static const hkVersionRegistry::ClassAction s_updateActions[] =
{
	// variants
	{ 0x6728e4b7, 0x6728e4b7, hkVersionRegistry::VERSION_VARIANT, "hkaBoneAttachment", HK_NULL },
	{ 0xf598a34e, 0xf598a34e, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainer", HK_NULL },
	{ 0x853a899c, 0x853a899c, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainerNamedVariant", HK_NULL },
	{ 0xdce3ca6b, 0xdce3ca6b, hkVersionRegistry::VERSION_VARIANT, "hkMemoryResourceHandle", HK_NULL },
	{ 0xbe6765dd, 0xbe6765dd, hkVersionRegistry::VERSION_VARIANT, "hkxMaterial", HK_NULL },
	{ 0x914da6c1, 0x914da6c1, hkVersionRegistry::VERSION_VARIANT, "hkxAttribute", HK_NULL },
	{ 0x1667c01c, 0x1667c01c, hkVersionRegistry::VERSION_VARIANT, "hkxAttributeGroup", HK_NULL },
	{ 0x06af1b5a, 0x06af1b5a, hkVersionRegistry::VERSION_VARIANT, "hkxNode", HK_NULL },
	{ 0x72e8e849, 0x72e8e849, hkVersionRegistry::VERSION_VARIANT, "hkxMesh", HK_NULL },
	{ 0x912c8863, 0x912c8863, hkVersionRegistry::VERSION_VARIANT, "hkxMeshSection", HK_NULL },
	{ 0x64e9a03c, 0x64e9a03c, hkVersionRegistry::VERSION_VARIANT, "hkxMeshUserChannelInfo", HK_NULL },
	{ 0x445a443a, 0x445a443a, hkVersionRegistry::VERSION_VARIANT, "hkxAttributeHolder", HK_NULL },
	{ 0x1388d601, 0x1388d601, hkVersionRegistry::VERSION_VARIANT, "hkCustomAttributesAttribute", HK_NULL },
	{ 0xbff19005, 0xbff19005, hkVersionRegistry::VERSION_VARIANT, "hkCustomAttributes", HK_NULL },
	{ 0x02ea23f0, 0x02ea23f0, hkVersionRegistry::VERSION_VARIANT, "hkxMaterialTextureStage", HK_NULL },
	{ 0x15d99dc6, 0x15d99dc6, hkVersionRegistry::VERSION_VARIANT, "hkbVariableValueSet", HK_NULL },
	{ 0xa71c409c, 0xd2963e7c, hkVersionRegistry::VERSION_VARIANT, "hkdShape", HK_NULL },
	{ 0x1bbfdb97, 0x1bbfdb97, hkVersionRegistry::VERSION_VARIANT, "hkdBody", HK_NULL },
	{ 0x8e9b1727, 0x8e9b1727, hkVersionRegistry::VERSION_HOMOGENEOUSARRAY, "hkxVertexBuffer", HK_NULL },

	// Common
	BINARY_IDENTICAL(0x38771f8e, 0x14425e51, "hkClass"), // changes in hkClassMember
	BINARY_IDENTICAL(0xa5240f57, 0x4a986551, "hkClassMember"), // added TYPE_HALF (COM-588), TYPE_STRINGPTR (COM-660)
	REMOVED("hkHalf"),
	{ 0x2a57a90a, 0xeb6e96e3, hkVersionRegistry::VERSION_COPY, "hkUiAttribute", HK_NULL },
	{ 0x3a27cba3, 0xd404a39a, hkVersionRegistry::VERSION_COPY, "hkArrayTypeAttribute", HK_NULL },

	// Physics
	REMOVED("hkpStabilizedBoxMotion"),
	REMOVED("hkpStabilizedSphereMotion"),
	
	// binary identical, hkHalf COM-588, if set to VERSION_COPY - you must copy hkHalf type members manually
	{ 0xf4fbf9b5, 0x2ca3e906, hkVersionRegistry::VERSION_MANUAL, "hkpStorageExtendedMeshShapeMaterial", Update_ignore },
	// binary identical, changes in hkpStorageExtendedMeshShapeMaterial, if set to VERSION_COPY - you must copy hkpStorageExtendedMeshShapeMaterial type members manually
	{ 0x0e0d8c23, 0xa7401420, hkVersionRegistry::VERSION_MANUAL, "hkpStorageExtendedMeshShapeMeshSubpartStorage", Update_ignore },
	// binary identical, changes in hkpStorageExtendedMeshShapeMaterial, if set to VERSION_COPY - you must copy hkpStorageExtendedMeshShapeMaterial type members manually
	{ 0x14d4585e, 0xd7628aa1, hkVersionRegistry::VERSION_MANUAL, "hkpStorageExtendedMeshShapeShapeSubpartStorage", Update_ignore },
	// member renames keep the object binary identical, changes in hkpMotion, if set to VERSION_COPY - you must copy hkpEntity members of hkpMotion type manually
	// however hkpMotion::MotionType enumeration changes also, and we need to handle that.
	{ 0x5a8169ee, 0xe7f760b1, hkVersionRegistry::VERSION_MANUAL, "hkpEntity", Update_hkpEntity },
	// hkpMotion::m_gravityFactor is binary identical, hkHalf COM-588, if set to VERSION_COPY - you must copy hkpMotion::m_gravityFactor manually
	{ 0xb9256deb, 0x0843f599, hkVersionRegistry::VERSION_MANUAL, "hkpMotion", Update_hkpMotion },
	// hkpConstraintInstance::m_uid added. It is a no-save member.
	{ 0xd0e73ea6, 0x418c7656, hkVersionRegistry::VERSION_COPY, "hkpConstraintInstance", Update_ignore },
	// hkpVehicleData has had a member renamed.
	BINARY_IDENTICAL(0x3749cfce, 0x173feb43, "hkpVehicleData"),
	// hkpVehicleWheelCollide has a new non-serialized member m_type.
	{ 0xbaf6940c, 0x4a50fcb, hkVersionRegistry::VERSION_COPY, "hkpVehicleWheelCollide", Update_ignore },
	// New member added.
	{ 0xba643b88, 0x2743fc9f, hkVersionRegistry::VERSION_COPY, "hkpWorldCinfo", HK_NULL },

	REMOVED("hkpEntityDeactivator"),
	REMOVED("hkpSpatialRigidBodyDeactivatorSample"),

	// Behavior
	{ 0x74d51a65, 0x3645587c, hkVersionRegistry::VERSION_COPY, "hkbBalanceModifier", Update_BalanceModifier },
	{ 0x67392073, 0xa759ddf6, hkVersionRegistry::VERSION_COPY, "hkbBalanceModifierStepInfo", Update_BalanceModifierStepInfo },
	{ 0x189cc895, 0xa42bce6f, hkVersionRegistry::VERSION_COPY, "hkbBalanceControllerModifier", HK_NULL }, 
	{ 0x3f756dd5, 0xaf837ec7, hkVersionRegistry::VERSION_COPY, "hkbBehaviorGraph", HK_NULL },
	{ 0x1e05e53,  0x95aca5d,  hkVersionRegistry::VERSION_COPY, "hkbBehaviorGraphData", HK_NULL }, 
	{ 0x77870625, 0x1d940b94, hkVersionRegistry::VERSION_COPY, "hkbBlenderGenerator", HK_NULL },
	{ 0xf9f192ae, 0x7a3fd2db, hkVersionRegistry::VERSION_COPY, "hkbBlenderGeneratorChild", HK_NULL },
	{ 0x5ffc7ff6, 0x877b80b5, hkVersionRegistry::VERSION_COPY, "hkbConstrainRigidBodyModifier", HK_NULL },
	{ 0xe814310a, 0xa4a30087, hkVersionRegistry::VERSION_COPY, "hkbCatchFallModifier", HK_NULL },
	{ 0x64f7d5a4, 0xe8d604a0, hkVersionRegistry::VERSION_COPY, "hkbCatchFallModifierHand", HK_NULL },
	{ 0xc837a59c, 0x6eba06ee, hkVersionRegistry::VERSION_COPY, "hkbCheckBalanceModifier", Update_CheckBalanceModifier },
	{ 0xad95c972, 0x66ff988e, hkVersionRegistry::VERSION_COPY, "hkbContext", HK_NULL },
	{ 0x84a96b43, 0x6ebb687b, hkVersionRegistry::VERSION_COPY, "hkbCharacterData", HK_NULL },
	{ 0x52bd5952, 0x0495b1aa, hkVersionRegistry::VERSION_COPY, "hkbCharacterStringData", HK_NULL },
	{ 0x64ca7109, 0x0211dd03, hkVersionRegistry::VERSION_COPY, "hkbDemoConfigCharacterInfo", HK_NULL },
	{ 0x69227730, 0xe2bd01ca, hkVersionRegistry::VERSION_COPY, "hkbDetectCloseToGroundModifier", HK_NULL }, 
	{ 0x278c3c95, 0xbde631dd, hkVersionRegistry::VERSION_COPY, "hkbEvaluateHandleModifier", HK_NULL },
	{ 0xbfa3dc93, 0x1f9d33f3, hkVersionRegistry::VERSION_COPY, "hkbFaceTargetModifier", HK_NULL },
	{ 0xba78ca90, 0xa35aa7f4, hkVersionRegistry::VERSION_COPY, "hkbFootIkModifier", HK_NULL },
	{ 0xbd8fae87, 0xe9c42c5d, hkVersionRegistry::VERSION_COPY, "hkbFootIkModifierLeg", HK_NULL },
	{ 0xf74661dd, 0x860eed6e, hkVersionRegistry::VERSION_COPY, "hkbHandIkModifier", HK_NULL },
	{ 0xe12f69f1, 0x5fc9a58,  hkVersionRegistry::VERSION_COPY, "hkbHandIkModifierHand", HK_NULL },
	{ 0xb84102a0, 0x0cf2dcc6, hkVersionRegistry::VERSION_COPY, "hkbGeneratorOutputTrackHeader", HK_NULL },
	{ 0xc1eebde6, 0xf47b781d, hkVersionRegistry::VERSION_COPY, "hkbGeneratorOutputTracks", HK_NULL },
	{ 0x4ddf213e, 0x6be843e0, hkVersionRegistry::VERSION_COPY, "hkbGeneratorTransitionEffect", HK_NULL },
	{ 0x509fafa2, 0x3065aa6d, hkVersionRegistry::VERSION_COPY, "hkbKeyframeBonesModifier", HK_NULL },
	{ 0x29257664, 0x5da10ed9, hkVersionRegistry::VERSION_COPY, "hkbMoveBoneTowardTargetModifier", HK_NULL },
	{ 0x7ed87d76, 0x8e2e192f, hkVersionRegistry::VERSION_COPY, "hkbPoweredRagdollModifier", HK_NULL },
	{ 0x0f79f135, 0x72deb7a6, hkVersionRegistry::VERSION_COPY, "hkbPoweredRagdollModifierKeyframeInfo", HK_NULL },
	{ 0xb427cccc, 0x689d3e34, hkVersionRegistry::VERSION_COPY, "hkbRagdollDriverModifier", HK_NULL },
	{ 0x3572bbc4, 0x76c4f6fd, hkVersionRegistry::VERSION_COPY, "hkbReachTowardTargetModifier", HK_NULL },
	{ 0xe762cb91, 0x5be561aa, hkVersionRegistry::VERSION_COPY, "hkbReachTowardTargetModifierHand", HK_NULL },
	{ 0x449e8371, 0x0ab8ae91, hkVersionRegistry::VERSION_COPY, "hkbSimpleCharacter", HK_NULL },
	{ 0x83c43a9a, 0xa9c76b33, hkVersionRegistry::VERSION_COPY, "hkbStateMachine", HK_NULL },
	{ 0x14a9d072, 0x6c40ed33, hkVersionRegistry::VERSION_COPY, "hkbStateMachineStateInfo", HK_NULL },
	{ 0x2d995d9c, 0xd1f819e1, hkVersionRegistry::VERSION_COPY, "hkbTarget", HK_NULL }, // HKF-772
	{ 0x8cd5c8b3, 0x6f1274cf, hkVersionRegistry::VERSION_COPY, "hkbTargetRigidBodyModifier", Update_ignore }, // HKF-772
	REMOVED("hkbClimbMountingCondition"),
	REMOVED("hkbCharacterBoneInfo"),

	// Destruction
	{ 0xd7ee252d, 0xcbc79e83, hkVersionRegistry::VERSION_COPY, "hkdDeformableBreakableShape", Update_hkdDeformableBreakableShape },
	{ 0xb57261de, 0xf8baedec, hkVersionRegistry::VERSION_COPY, "hkdCompoundBreakableShape", HK_NULL },
	{ 0xbe975804, 0xab8e7796, hkVersionRegistry::VERSION_COPY, "hkdBreakableShape", HK_NULL },
	{ 0xfce05ec9, 0x55deec85, hkVersionRegistry::VERSION_COPY, "hkdBreakableShapeConnection", HK_NULL },
	{ 0x7a38ac49, 0xb5c49f45, hkVersionRegistry::VERSION_COPY, "hkdFlexibleJointController", HK_NULL },
	{ 0x1dfa181d, 0x7fb3aae0, hkVersionRegistry::VERSION_COPY, "hkdShapeInstanceInfo", HK_NULL },
	{ 0xfec01075, 0x3024b49, hkVersionRegistry::VERSION_COPY, "hkdShapeInstanceInfoRuntimeInfo", HK_NULL },
	{ 0x4d607160, 0x2f67a6dd, hkVersionRegistry::VERSION_COPY, "hkdWoodFracture", HK_NULL },
	{ 0xc0d46386, 0xe40b6528, hkVersionRegistry::VERSION_COPY, "hkdWoodFractureSplittingData", HK_NULL },
	{ 0xbef2be0c, 0x297a264f, hkVersionRegistry::VERSION_COPY, "hkdDebrisFracture", HK_NULL },
	{ 0xf696cd0b, 0x175c2a93, hkVersionRegistry::VERSION_COPY, "hkdFracture", HK_NULL },
	{ 0xda8c7d7d, 0x7d4c7b72, hkVersionRegistry::VERSION_COPY, "hkdAction", HK_NULL },
	{ 0x515f10b1, 0xfe812f2d, hkVersionRegistry::VERSION_COPY, "hkdDestructionDemoConfig", HK_NULL },
	REMOVED("hkdBreakableBodyBlueprint"),

	{ 0, 0, 0, HK_NULL, HK_NULL }
};	 

static const hkVersionRegistry::ClassRename s_renames[] =
{
	{ "hkpVehicleRaycastWheelCollide", "hkpVehicleRayCastWheelCollide" },
	{ "hkpRejectRayChassisListener", "hkpRejectChassisListener" },
	{ HK_NULL, HK_NULL }
};

#define HK_COMPAT_VERSION_FROM hkHavok650r1Classes
#define HK_COMPAT_VERSION_TO hkHavok660b1Classes
#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO

} // namespace hkCompat_hk650r1_hk660b1

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
