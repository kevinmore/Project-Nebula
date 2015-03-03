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
#include <Common/Base/Container/BitField/hkBitField.h>

#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>

namespace hkCompat_hk510r1_hk550b1
{

#define REMOVED(TYPE) { 0,0, hkVersionRegistry::VERSION_REMOVED, TYPE, HK_NULL }
#define BINARY_IDENTICAL(OLDSIG,NEWSIG,TYPE) { OLDSIG, NEWSIG, hkVersionRegistry::VERSION_MANUAL, TYPE, HK_NULL }

	static void ObjectPacking( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor oldTransform( oldObj, "transform" );
		hkClassMemberAccessor newTranslation( newObj, "translation" );
		hkClassMemberAccessor newRotation( newObj, "rotation" );

		newTranslation.asVector4() = oldTransform.asTransform().t;

		hkRotation rotation;
		hkVector4* rotationMatrix = reinterpret_cast<hkVector4*>(oldTransform.asTransform().r);
		rotation.setRows(rotationMatrix[0], rotationMatrix[1], rotationMatrix[2]);
		hkQuaternion quaternion(rotation);
		newRotation.asVector4() = *(reinterpret_cast<hkClassMemberAccessor::Vector4*>(&quaternion.m_vec));
	}

	static void hkExtendedMeshSubpartsArray(
		hkVariant& oldObj,
		hkVariant& newObj,
		hkObjectUpdateTracker& tracker)
	{
		const hkClassMemberAccessor& oldObjMem = hkClassMemberAccessor(oldObj, "shapesSubparts");
		const hkClassMemberAccessor& newObjMem = hkClassMemberAccessor(newObj, "shapesSubparts");
		hkClassMemberAccessor::SimpleArray& oldArray = oldObjMem.asSimpleArray();
		hkClassMemberAccessor::SimpleArray& newArray = newObjMem.asSimpleArray();
		HK_ASSERT( 0xad78555b, oldArray.size == newArray.size );
		hkVariant oldVariant = {HK_NULL, &oldObjMem.object().getClass()};
		hkVariant newVariant = {HK_NULL, &newObjMem.object().getClass()};
		hkInt32 oldSize = oldVariant.m_class->getObjectSize();
		hkInt32 newSize = newVariant.m_class->getObjectSize();
		for( int i = 0; i < oldArray.size; ++i )
		{
			oldVariant.m_object = static_cast<char*>(oldArray.data) + oldSize*i;
			newVariant.m_object = static_cast<char*>(newArray.data) + newSize*i;
			ObjectPacking(oldVariant, newVariant, tracker);
		}
	}

	static void Update_hkaAnimationBinding( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// Handle the renamed m_animationTrackToBoneIndices to m_transformTrackToBoneIndices
		{
			hkClassMemberAccessor oldMember( oldObj, "animationTrackToBoneIndices" );
			hkClassMemberAccessor newMember( newObj, "transformTrackToBoneIndices" );
			newMember.asSimpleArray() = oldMember.asSimpleArray();
		}
	}

	static void Update_hkaSkeletalAnimation( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// Handle the renamed m_numberOfTracks to m_numberOfTransformTracks
		{
			hkClassMemberAccessor oldMember( oldObj, "numberOfTracks" );
			hkClassMemberAccessor newMember( newObj, "numberOfTransformTracks" );
			newMember.asInt32() = oldMember.asInt32();
		}
	}

	static void Update_ignore( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
	}

	static void Update_hkbCatchFallModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// Conversion from hkbEvent to hkInt32( event id )
		{
			hkClassMemberAccessor oldMember( oldObj, "catchFallDoneEvent" );
			hkClassMemberAccessor newMember( newObj, "catchFallDoneEventId" );

			hkClassMemberAccessor oldMemberSubMember = oldMember.member( "id" );

			newMember.asInt32() = oldMemberSubMember.asInt32();
		}

		// Member variable name change
		{
			hkClassMemberAccessor oldMember( oldObj, "catchFallDirectionRagdollBone" );
			hkClassMemberAccessor newMember( newObj, "catchFallDirectionRagdollBoneIndex" );

			newMember.asInt16() = oldMember.asInt16();
		}

		// Member variable type change from int to hkInt16
		{
			hkClassMemberAccessor oldMember( oldObj, "velocityRagdollBoneIndex" );
			hkClassMemberAccessor newMember( newObj, "velocityRagdollBoneIndex" );

			newMember.asInt16() = oldMember.asInt16();
		}

		// Member variable type change from int to hkInt16
		{
			hkClassMemberAccessor oldMember( oldObj, "raycastLayer" );
			hkClassMemberAccessor newMember( newObj, "raycastLayer" );

			newMember.asInt16() = oldMember.asInt16();
		}

		// Conversion from m_handIndex[MAX_LIMBS] to two hand structures
		{
			hkClassMemberAccessor oldMember( oldObj, "handIndex" );
			hkInt16* oldData = reinterpret_cast<hkInt16*>(oldMember.getAddress());

			hkClassMemberAccessor newMember1( newObj, "leftHand" );
			hkClassMemberAccessor newMember1SubMember1 = newMember1.member( "handIndex" );
			hkClassMemberAccessor newMember1SubMember2 = newMember1.member( "handIkTrackIndex" );
			hkClassMemberAccessor newMember1SubMember3 = newMember1.member( "animShoulderIndex" );
			hkClassMemberAccessor newMember1SubMember4 = newMember1.member( "ragdollShoulderIndex" );
			hkClassMemberAccessor newMember1SubMember5 = newMember1.member( "ragdollAnkleIndex" );

			newMember1SubMember1.asInt16() = newMember1SubMember2.asInt16() = *oldData;

			// This is required because currently it seems that new embedded structures defaults are not applied.
			newMember1SubMember3.asInt16() = newMember1SubMember4.asInt16() = newMember1SubMember5.asInt16() = -1;

			oldData++;

			hkClassMemberAccessor newMember2( newObj, "rightHand" );
			hkClassMemberAccessor newMember2SubMember1 = newMember2.member( "handIndex" );
			hkClassMemberAccessor newMember2SubMember2 = newMember2.member( "handIkTrackIndex" );
			hkClassMemberAccessor newMember2SubMember3 = newMember2.member( "animShoulderIndex" );
			hkClassMemberAccessor newMember2SubMember4 = newMember2.member( "ragdollShoulderIndex" );
			hkClassMemberAccessor newMember2SubMember5 = newMember2.member( "ragdollAnkleIndex" );

			newMember2SubMember1.asInt16() = newMember2SubMember2.asInt16() = *oldData;

			// This is required because currently it seems that new embedded structures defaults are not applied.
			newMember2SubMember3.asInt16() = newMember2SubMember4.asInt16() = newMember2SubMember5.asInt16() = -1;

		}
	}

	static void Update_hkbHandIkModifierHandInternal( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldMember( oldObj, "handIndex" );
		hkClassMemberAccessor newMember( newObj, "handIkTrackIndex" );

		newMember.asInt16() = oldMember.asInt16();
	}

	static void Update_hkbHandIkModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldHands( oldObj, "hands" );
		hkClassMemberAccessor newHands( newObj, "hands" );

		const int numHands = oldHands.asSimpleArray().size;

		hkVariant oldHandVariant;
		hkVariant newHandVariant;

		oldHandVariant.m_class = &oldHands.object().getClass();
		newHandVariant.m_class = &newHands.object().getClass();

		int oldHandStride = oldHands.getClassMember().getStructClass().getObjectSize();
		int newHandStride = newHands.getClassMember().getStructClass().getObjectSize();

		for( int i = 0; i < numHands; ++i )
		{
			oldHandVariant.m_object = static_cast<char*>(oldHands.asSimpleArray().data) + i * oldHandStride;
			newHandVariant.m_object = static_cast<char*>(newHands.asSimpleArray().data) + i * newHandStride;

			Update_hkbHandIkModifierHandInternal( oldHandVariant, newHandVariant, tracker );
		}
	}

	static void Update_hkbHandIkModifierHand( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		HK_ERROR( 0x2d9a6f3e, "This function should never be called." );
	}

	static void Update_hkbJigglerGroup( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		{
			hkClassMemberAccessor oldMember( oldObj, "maxBoneLengthFraction" );
			hkClassMemberAccessor newMember( newObj, "maxElongation" );

			newMember.asReal() = oldMember.asReal();
		}

		{
			hkClassMemberAccessor oldMember( oldObj, "minBoneLengthFraction" );
			hkClassMemberAccessor newMember( newObj, "maxCompression" );

			newMember.asReal() = oldMember.asReal();
		}
	}

static const hkVersionRegistry::ClassAction s_updateActions[] =
{
	// common
	{ 0x8bdd3e9a, 0x8bdd3e9a, hkVersionRegistry::VERSION_VARIANT, "hkaBoneAttachment", HK_NULL },
	{ 0xf598a34e, 0xf598a34e, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainer", HK_NULL },
	{ 0x853a899c, 0x853a899c, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainerNamedVariant", HK_NULL },
	{ 0xf2ec0c9c, 0xf2ec0c9c, hkVersionRegistry::VERSION_VARIANT, "hkxMaterial", HK_NULL },
	{ 0x914da6c1, 0x914da6c1, hkVersionRegistry::VERSION_VARIANT, "hkxAttribute", HK_NULL },
	{ 0x1667c01c, 0x1667c01c, hkVersionRegistry::VERSION_VARIANT, "hkxAttributeGroup", HK_NULL },
	{ 0xf65b406a, 0x44ff20d2, hkVersionRegistry::VERSION_COPY, "hkpConvexVerticesShape", HK_NULL },
	{ 0x06af1b5a, 0x06af1b5a, hkVersionRegistry::VERSION_VARIANT, "hkxNode", HK_NULL },
	{ 0xe085ba9f, 0xe085ba9f, hkVersionRegistry::VERSION_VARIANT, "hkxMaterialTextureStage", HK_NULL },
	{ 0x72e8e849, 0x72e8e849, hkVersionRegistry::VERSION_VARIANT, "hkxMesh", HK_NULL },
	{ 0x912c8863, 0x912c8863, hkVersionRegistry::VERSION_VARIANT, "hkxMeshSection", HK_NULL },
	{ 0x64e9a03c, 0x64e9a03c, hkVersionRegistry::VERSION_VARIANT, "hkxMeshUserChannelInfo", HK_NULL },
	{ 0x445a443a, 0x445a443a, hkVersionRegistry::VERSION_VARIANT, "hkxAttributeHolder", HK_NULL },
	{ 0x1388d601, 0x1388d601, hkVersionRegistry::VERSION_VARIANT, "hkCustomAttributesAttribute", HK_NULL },
	{ 0xbff19005, 0xbff19005, hkVersionRegistry::VERSION_VARIANT, "hkCustomAttributes", HK_NULL },
	{ 0x57061454, 0x57061454, hkVersionRegistry::VERSION_HOMOGENEOUSARRAY, "hkxVertexBuffer", HK_NULL },

	// base
	BINARY_IDENTICAL(0xfc41dc67, 0x38771f8e, "hkClass"), // changes in hkClassMember and hkClassEnum, COM-228
	BINARY_IDENTICAL(0x258a78ee, 0xa5240f57, "hkClassMember"), // +serialized(false) for hkClassMember::m_attributes, COM-228
	BINARY_IDENTICAL(0x528ce1e5, 0x8a3609cf, "hkClassEnum"), // +serialized(false) for hkClassMember::m_attributes, COM-228

	// physics
	BINARY_IDENTICAL(0xeb9edbdc, 0x33f74135, "hkpBallAndSocketConstraintDataAtoms"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0xe45bdb74, 0x77aabbef, "hkpBallAndSocketConstraintData"), // changes in hkpConstraintAtom/hkpConstraintData
	BINARY_IDENTICAL(0x8c6fb8d3, 0xea22d4f9, "hkpBallSocketChainData"), // changes in hkpConstraintAtom/hkpConstraintData
	BINARY_IDENTICAL(0x8ba3b296, 0x98900a8a, "hkpBreakableConstraintData"), // changes in hkpConstraintAtom/hkpConstraintData
	BINARY_IDENTICAL(0x75b4a341, 0x71423c29, "hkpBridgeAtoms"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x4a9bffad, 0x033eab5e, "hkpGenericConstraintData"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x5e4166c4, 0x8c4d3cf6, "hkpHingeConstraintDataAtoms"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x05671ba6, 0x502cad5a, "hkpHingeConstraintData"), // changes in hkpConstraintData
	BINARY_IDENTICAL(0x0381a5af, 0x19a836b1, "hkpHingeLimitsDataAtoms"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x0b582c21, 0xd023da1a, "hkpHingeLimitsData"), // changes in hkpConstraintData
	BINARY_IDENTICAL(0xbebe526b, 0xeb91c599, "hkpLimitedHingeConstraintDataAtoms"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0xd3afee7a, 0xe863aa21, "hkpLimitedHingeConstraintData"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x290c3390, 0x73937118, "hkpMalleableConstraintData"), // changes in hkpConstraintAtom/hkpConstraintData
	BINARY_IDENTICAL(0x65380e91, 0xcf3f2f29, "hkpPointToPathConstraintData"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x25effb78, 0xb191e7d3, "hkpPointToPlaneConstraintDataAtoms"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0xd23f1309, 0xb3c7ee7f, "hkpPointToPlaneConstraintData"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0xb978a5ec, 0x2c90d2b4, "hkpPoweredChainData"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x1666c241, 0xf7809c2d, "hkpPrismaticConstraintDataAtoms"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x9a4b7b0d, 0xcf55cbd3, "hkpPrismaticConstraintData"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0xa457c77b, 0xb0b4ce17, "hkpPulleyConstraintDataAtoms"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0xb380e4b9, 0x8f4fcde2, "hkpPulleyConstraintData"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x054c1db0, 0xbcca03b0, "hkpRagdollConstraintDataAtoms"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x802f9dfd, 0xeb0cd053, "hkpRagdollConstraintData"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x92e9ef7c, 0xf4de43f4, "hkpRagdollLimitsDataAtoms"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x50801cf8, 0x93b97b48, "hkpRagdollLimitsData"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0xea09782c, 0xab648d8d, "hkpRotationalConstraintDataAtoms"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0xb8b90fed, 0xec1fc462, "hkpRotationalConstraintData"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x496cceaa, 0x28c7f0f3, "hkpSerializedAgentNnEntry"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x7aef1198, 0x330795be, "hkpStiffSpringChainData"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x10d9496c, 0x4616cbb4, "hkpStiffSpringConstraintDataAtoms"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x03c66e67, 0x808a3957, "hkpStiffSpringConstraintData"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0xeda6f2c8, 0xdcf27bc0, "hkpWheelConstraintDataAtoms"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x0ae5d09b, 0x94415572, "hkpWheelConstraintData"), // changes in hkpConstraintAtom
	BINARY_IDENTICAL(0x357781ee, 0xed982d04, "hkpConstraintAtom"), // added TYPE_MODIFIER_IGNORE_CONSTRAINT enum value, changed value for TYPE_MAX, HVK-3031
	BINARY_IDENTICAL(0xf9515b8a, 0xc18f0d87, "hkpConstraintData"), // Added a new constraint data type HVK-4174
	{ 0x137ad522, 0xfce016a3, hkVersionRegistry::VERSION_COPY, "hkpExtendedMeshShapeShapesSubpart", ObjectPacking},
	{ 0x3f87178f, 0x260c3908, hkVersionRegistry::VERSION_COPY, "hkpExtendedMeshShape", hkExtendedMeshSubpartsArray},
	{ 0xfb086d6b, 0x3fcd7295, hkVersionRegistry::VERSION_COPY, "hkpWorldCinfo", HK_NULL },
	{ 0x250ee68f, 0x60960336, hkVersionRegistry::VERSION_COPY, "hkpCollisionFilter", HK_NULL }, // Fixed alignment for ps2 COM-181, COM-294
	BINARY_IDENTICAL(0xcf2fe779, 0x358bfe9c, "hkpVehicleSuspensionSuspensionWheelParameters"), // rename members, HVK-1890
	BINARY_IDENTICAL(0x35a4d7b2, 0xaf5056fa, "hkpVehicleSuspension"), // changes in hkpVehicleSuspensionSuspensionWheelParameters
	BINARY_IDENTICAL(0x8d5654ee, 0x80ce3610, "hkpVehicleInstanceWheelInfo"), // rename members, HVK-1890
	BINARY_IDENTICAL(0x232a16b3, 0x22c896d9, "hkpVehicleInstance"), // changes in hkpVehicleInstanceWheelInfo
	REMOVED("hkpMoppEmbeddedShape"),
	REMOVED("hkpPackedConvexVerticesShape"),
	REMOVED("hkpPackedConvexVerticesShapeFourVectors"),
	REMOVED("hkpPackedConvexVerticesShapeVector4IntW"),

	// behavior
	{ 0xa4253200, 0x86fa3de7, hkVersionRegistry::VERSION_COPY, "hkbBehaviorGraph", HK_NULL },
	{ 0xbba34f99, 0xc99c5164, hkVersionRegistry::VERSION_COPY, "hkbBlenderGeneratorChild", HK_NULL },
	{ 0xd52c6d90, 0x5b49cdd5, hkVersionRegistry::VERSION_COPY, "hkbBlendingTransitionEffect", HK_NULL },
	{ 0xf606246c, 0x1baa90d7, hkVersionRegistry::VERSION_COPY, "hkbCharacter", Update_ignore },				// these don't need to be versioned
	{ 0xecd5a299, 0x191d627d, hkVersionRegistry::VERSION_COPY, "hkbClimbMountingPredicate", HK_NULL },
	{ 0x4333653e, 0x2fafbf05, hkVersionRegistry::VERSION_COPY, "hkbContext", HK_NULL },
	{ 0xbd247645, 0xc93ae059, hkVersionRegistry::VERSION_COPY, "hkbCatchFallModifier", Update_hkbCatchFallModifier },
	{ 0x44d86267, 0xa10bf96a, hkVersionRegistry::VERSION_COPY, "hkbGeneratorOutput", Update_ignore },		// these don't need to be versioned
	{ 0x978d2a63, 0x504fa563, hkVersionRegistry::VERSION_COPY, "hkbGeneratorOutputTrack", Update_ignore },	// these don't need to be versioned
	{ 0x95ff3258, 0xf5264fd4, hkVersionRegistry::VERSION_COPY, "hkbGeneratorTransitionEffect", HK_NULL },
	{ 0x2b2f4ff,  0xc92963cf, hkVersionRegistry::VERSION_COPY, "hkbHandIkModifier", Update_hkbHandIkModifier },
	{ 0xfd01aaf,  0x5e7f276b, hkVersionRegistry::VERSION_COPY, "hkbHandIkModifierHand", Update_hkbHandIkModifierHand },
	{ 0x492e39c9, 0xfe5e54b6, hkVersionRegistry::VERSION_COPY, "hkbJigglerGroup", Update_hkbJigglerGroup },
	{ 0xefef656e, 0xa13ac125, hkVersionRegistry::VERSION_COPY, "hkbMirrorModifier", HK_NULL },
	{ 0xb7422e1a, 0x8f3e0019, hkVersionRegistry::VERSION_COPY, "hkbRotateCharacterModifier", HK_NULL },
	{ 0xfa94f179, 0xe4df25a6, hkVersionRegistry::VERSION_COPY, "hkbSequence", HK_NULL },
	{ 0x933b54af, 0x622ff0bf, hkVersionRegistry::VERSION_COPY, "hkbTarget", HK_NULL },
	{ 0x56c09f57, 0xd5ad3f4e, hkVersionRegistry::VERSION_COPY, "hkbTimerModifier", HK_NULL },
	{ 0xe25d0a56, 0x62eabf22, hkVersionRegistry::VERSION_COPY, "hkbDemoConfig", HK_NULL }, // added m_numTracks member, HKF-537
	REMOVED("hkbCharacterFakeQueue"),

	// animation
	{ 0xe39df839, 0xfb496074, hkVersionRegistry::VERSION_COPY, "hkaAnimationBinding", Update_hkaAnimationBinding },
	{ 0x2b2e784a, 0x3e2d394f, hkVersionRegistry::VERSION_COPY, "hkaInterleavedSkeletalAnimation", Update_hkaSkeletalAnimation },
	{ 0x5b9ff2db, 0xb52635c4, hkVersionRegistry::VERSION_COPY, "hkaSkeletalAnimation", Update_hkaSkeletalAnimation },
	{ 0xa35e6164, 0x334dbe6c, hkVersionRegistry::VERSION_COPY, "hkaSkeleton", HK_NULL },
	{ 0xd338d9e7, 0x27c6cafa, hkVersionRegistry::VERSION_COPY, "hkaWaveletSkeletalAnimationCompressionParams", HK_NULL },

	{ 0, 0, 0, HK_NULL, HK_NULL }
};

static const hkVersionRegistry::ClassRename s_renames[] =
{
	{ HK_NULL, HK_NULL }
};

#define HK_COMPAT_VERSION_FROM hkHavok510r1Classes
#define HK_COMPAT_VERSION_TO hkHavok550b1Classes
#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO
} // namespace hkCompat_hk510r1_hk550b1

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
