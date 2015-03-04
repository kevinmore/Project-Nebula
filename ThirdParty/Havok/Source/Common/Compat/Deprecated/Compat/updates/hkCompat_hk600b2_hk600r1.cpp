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

namespace hkCompat_hk600b2_hk600r1
{
#define REMOVED(TYPE) { 0,0, hkVersionRegistry::VERSION_REMOVED, TYPE, HK_NULL }
#define BINARY_IDENTICAL(OLDSIG,NEWSIG,TYPE) { OLDSIG, NEWSIG, hkVersionRegistry::VERSION_MANUAL, TYPE, HK_NULL }

	static void Update_ignore( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
	}

	static void Update_hkbMirroredSkeletonInfo( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// Nothing to do here, no new variables have been added, only old ones removed.
		// Note that m_mirrorAxis now has a new interpretation.  The new meaning cannot be derived from the old meaning.
		// In compensation, the new mirroring is much much easier to setup, hopefully shouldn't be too big of a burden.
	}

	static void Update_hkbFootIkGainsInternal( hkVariant& oldObj, hkVariant& newObj )
	{
		// changed member name
		hkClassMemberAccessor oldMember(oldObj, "worldFromModelFeedbackUpDownBias");
		hkClassMemberAccessor newMember(newObj, "errorUpDownBias");

		newMember.asReal() = oldMember.asReal();
	}

	static void Update_hkbFootIkModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// version the gains
		{
			hkClassMemberAccessor oldMember(oldObj, "gains");
			hkClassMemberAccessor newMember(newObj, "gains");

			hkVariant oldVariant;
			oldVariant.m_class = &(oldMember.getClassMember().getStructClass());
			oldVariant.m_object = oldMember.getAddress();

			hkVariant newVariant;
			newVariant.m_class = &(newMember.getClassMember().getStructClass());
			newVariant.m_object = newMember.getAddress();

			Update_hkbFootIkGainsInternal( oldVariant, newVariant );
		}
	}

	static void Update_hkbPoweredRagdollModifierKeyframeInfo( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// changed from hkVector4 to hkQuaternion
		hkClassMemberAccessor oldRotation(oldObj, "keyframedRotation");
		hkClassMemberAccessor newRotation(newObj, "keyframedRotation");

		newRotation.asVector4() = oldRotation.asVector4();
	}

	static void Update_hkbPositionRelativeSelectorGenerator( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// changed from hkVector4 to hkQuaternion
		hkClassMemberAccessor oldRotation(oldObj, "targetRotation");
		hkClassMemberAccessor newRotation(newObj, "targetRotation");

		newRotation.asVector4() = oldRotation.asVector4();
	}

	static void Update_hkbMoveBoneTowardTargetModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// changed from hkVector4 to hkQuaternion
		hkClassMemberAccessor oldRotation(oldObj, "currentBoneRotationOut");
		hkClassMemberAccessor newRotation(newObj, "currentBoneRotationOut");

		newRotation.asVector4() = oldRotation.asVector4();
	}

	static void Update_hkdSliceFracture( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// slice thickness becomes number of parts
		hkClassMemberAccessor oldMember(oldObj, "sliceThickness");
		hkClassMemberAccessor newMember(newObj, "numSubparts");
		newMember.asReal() = 1.0f / oldMember.asReal();
	}

	static void Update_hkdDeformationController( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkVersionUtil::renameMember( oldObj, "deformationRestitution", newObj, "softness");
	}

	static void Update_hclConvexHeightFieldShape( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// Re-quantize heights
		{
			hkClassMemberAccessor oldMember(oldObj, "heights");
			hkClassMemberAccessor newMember(newObj, "heights");
			HK_ASSERT(0x3a663d4b, oldMember.getClassMember().getSubType()==hkClassMember::TYPE_UINT32);
			HK_ASSERT(0x107bf811, newMember.getClassMember().getSubType()==hkClassMember::TYPE_UINT8);

			hkClassMemberAccessor::SimpleArray& heightsOld = oldMember.asSimpleArray();
			hkClassMemberAccessor::SimpleArray& heightsNew = newMember.asSimpleArray();

			const int numHeights = heightsOld.size;
			heightsNew.size = numHeights;

			hkUint32* heightsOldData = reinterpret_cast<hkUint32*> (heightsOld.data);
			hkUint8* heightsNewData = reinterpret_cast<hkUint8*> (heightsOldData); // We reuse the memory as we need less space than before

			const hkReal quantize32bit = 1.0f/hkReal(0xffffffff);

			for (int h=0; h<numHeights; ++h)
			{
				const hkReal floatValue = heightsOldData[h] * quantize32bit;
				heightsNewData[h] = static_cast<hkUint8> (floatValue * 255.0f);
			}
		}

		// Fix the localMapToScale
		{
			hkClassMemberAccessor oldMember (oldObj, "localToMapScale");
			hkClassMemberAccessor newMember (newObj, "localToMapScale");

			const hkReal scale =  hkReal(0xff) / hkReal(0xffffffff);
			newMember.asVector4().r[3] = oldMember.asVector4().r[3] * scale;

		}
	}

	// LocalRange entries as in 6.0.0
	struct LocalRangeEntry
	{
		hkUint32 m_particleIndex;
		hkUint32 m_referenceVertex; 
		hkReal m_maximumDistance;
		hkReal m_maxNormalDistance; 
		hkReal m_minNormalDistance; 
	};

	// HCL-533
	static HK_FORCE_INLINE hkBool constraintLess( const LocalRangeEntry& a, const LocalRangeEntry& b )
	{
		return a.m_referenceVertex < b.m_referenceVertex;

	}

	static void Update_hclLocalRangeConstraintSet( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// HCL-533 - Entries must be ordered by reference index
		hkClassMemberAccessor oldMember (oldObj, "localConstraints");
		hkClassMemberAccessor newMember (newObj, "localConstraints");

		hkClassMemberAccessor::SimpleArray& arrayOld = oldMember.asSimpleArray();
		hkClassMemberAccessor::SimpleArray& arrayNew = newMember.asSimpleArray();

		arrayNew = arrayOld; // Reuse the same memory

		LocalRangeEntry* entries = reinterpret_cast<LocalRangeEntry*> (arrayNew.data);

		// HCL-533 - We need to reorder by "referenceVertex" in order to execute on SPU
		hkSort(entries, arrayNew.size, constraintLess );
	}

static const hkVersionRegistry::ClassAction s_updateActions[] =
{
	// common
	{ 0x6728e4b7, 0x6728e4b7, hkVersionRegistry::VERSION_VARIANT, "hkaBoneAttachment", HK_NULL },
	{ 0xf598a34e, 0xf598a34e, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainer", HK_NULL },
	{ 0x853a899c, 0x853a899c, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainerNamedVariant", HK_NULL },
	{ 0x3d4223b3, 0x3d4223b3, hkVersionRegistry::VERSION_VARIANT, "hkMemoryResourceHandle", HK_NULL },
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
	{ 0x8e9b1727, 0x8e9b1727, hkVersionRegistry::VERSION_HOMOGENEOUSARRAY, "hkxVertexBuffer", HK_NULL },

	// physics
	{ 0xac084729, 0xdbf952ec, hkVersionRegistry::VERSION_MANUAL, "hkAabbUint32", Update_ignore }, // HVK-4627
	{ 0xe2f64d5c, 0x4793379a, hkVersionRegistry::VERSION_COPY, "hkpEntity", HK_NULL }, // HKF-735

	{ 0x23917cac, 0x85228673, hkVersionRegistry::VERSION_COPY, "hkpExtendedMeshShape", HK_NULL }, // HVK-4093
	{ 0xd42fac97, 0xdc8a0bca, hkVersionRegistry::VERSION_COPY, "hkpExtendedMeshShapeShapesSubpart" , HK_NULL }, // HVK-4093

	REMOVED("hkpHandle"), // HKF-735
	REMOVED("hkpHandleCollection"), // HKF-735
	{ 0x1318005c, 0x782e8ff3, hkVersionRegistry::VERSION_COPY, "hkpListShape", HK_NULL }, // HVK-4643
	BINARY_IDENTICAL( 0x6d1dc26a, 0x80df0f90, "hkpListShapeChildInfo" ),

	// animation
	{ 0x334dbe6c, 0x2a1e146f, hkVersionRegistry::VERSION_COPY, "hkaSkeleton", HK_NULL }, // HKF-735

	// behavior
	REMOVED("hkbAttachmentModifierAttachmentProperties"),
	{ 0x86028e60, 0x15d7d9d8, hkVersionRegistry::VERSION_COPY, "hkbAttachmentSetup", HK_NULL },
	{ 0x257691a0, 0xece659ff, hkVersionRegistry::VERSION_COPY, "hkbAttachmentModifier", HK_NULL },
	{ 0x1e9bec06, 0xc16e9ae8, hkVersionRegistry::VERSION_COPY, "hkbBehaviorGraph", HK_NULL },
	{ 0xf927a830, 0xda8c7d7d, hkVersionRegistry::VERSION_COPY, "hkbCharacter", HK_NULL }, // not really versioned because it is only for HBT
	{ 0x2fafbf05, 0xad95c972, hkVersionRegistry::VERSION_COPY, "hkbContext", HK_NULL },
	{ 0xd51f8ab3, 0xc7d6083,  hkVersionRegistry::VERSION_COPY, "hkbDelayedModifier", HK_NULL },
	REMOVED("hkbDetectMouseSpringModifier"),
	BINARY_IDENTICAL( 0x4f6a5aec, 0x91bfa071, "hkbFootIkGains" ),
	BINARY_IDENTICAL( 0xe49e625e, 0x71452ee7, "hkbFootIkControlData" ),
	BINARY_IDENTICAL( 0x713555b8, 0x86e4783c, "hkbFootIkControlsModifier" ),
	{ 0x78a6ee24, 0x96eb0e18, hkVersionRegistry::VERSION_COPY, "hkbFootIkModifier", Update_hkbFootIkModifier },
	BINARY_IDENTICAL( 0x8c68ad11, 0x6eabe974, "hkbGeneratorOutput" ),
	REMOVED("hkbKeyframeData"),
	{ 0xeb4e3bbd, 0xc6c2da4f, hkVersionRegistry::VERSION_COPY, "hkbMirroredSkeletonInfo", Update_hkbMirroredSkeletonInfo },
	// changed hkVector4 to hkQuaternion, which is binary identical but you need an update function
	{ 0x9e27ca98, 0x4dc78b75, hkVersionRegistry::VERSION_COPY, "hkbMoveBoneTowardTargetModifier", Update_hkbMoveBoneTowardTargetModifier },
	// changed hkVector4 to hkQuaternion and added members
	{ 0x87579894, 0xd941d03e, hkVersionRegistry::VERSION_COPY, "hkbPositionRelativeSelectorGenerator", Update_hkbPositionRelativeSelectorGenerator },
	// binary identical, but needs an update function
	{ 0x9f4d8c93, 0x558db289, hkVersionRegistry::VERSION_COPY, "hkbPoweredRagdollModifier", Update_ignore },
	// changed hkVector4 to hkQuaternion and added members
	{ 0x58f68fc4, 0x0f79f135, hkVersionRegistry::VERSION_COPY, "hkbPoweredRagdollModifierKeyframeInfo", Update_hkbPoweredRagdollModifierKeyframeInfo },
	{ 0x56e6c995, 0x74da13d7, hkVersionRegistry::VERSION_COPY, "hkbSenseHandleModifier", HK_NULL }, // HKF-735
	{ 0xabacb11c, 0xd8e1976b, hkVersionRegistry::VERSION_COPY, "hkbStateMachine", HK_NULL },
	{ 0x64333cca, 0x8359bf76, hkVersionRegistry::VERSION_COPY, "hkbTargetRigidBodyModifier", HK_NULL },
	{ 0x7d6e4cea, 0x21898fef, hkVersionRegistry::VERSION_COPY, "hkbVariableBindingSet", HK_NULL },
	{ 0xfc005983, 0x0487a360, hkVersionRegistry::VERSION_COPY, "hkbVariableBindingSetBinding", HK_NULL },


	// cloth
	REMOVED("hclPlanarHeightFieldShape"),
	{ 0xbc4f1b72, 0xc01dc33a, hkVersionRegistry::VERSION_COPY, "hclVolumeConstraintFrameData", HK_NULL},
	{ 0xa0696629, 0x1264c5e2, hkVersionRegistry::VERSION_COPY, "hclVolumeConstraintApplyData", HK_NULL},
	{ 0xd3841c40, 0xc5102320, hkVersionRegistry::VERSION_COPY, "hclVolumeConstraint", HK_NULL},
	{ 0xe2623674, 0x9d3ef807, hkVersionRegistry::VERSION_COPY, "hclMeshMeshDeformOperator", HK_NULL},
	{ 0x4c062775, 0x39444400, hkVersionRegistry::VERSION_COPY, "hclMeshMeshDeformOperatorTriangleVertexPair", HK_NULL},
	{ 0x1d3b89c7, 0xdf4799d0, hkVersionRegistry::VERSION_MANUAL, "hclConvexHeightFieldShape", Update_hclConvexHeightFieldShape},
	{ 0x3e768530, 0xe641e54a, hkVersionRegistry::VERSION_COPY, "hclVertexFloatInput", HK_NULL},
	{ 0xefe0ad15, 0x28b2a471, hkVersionRegistry::VERSION_MANUAL, "hclLocalRangeConstraintSet", Update_hclLocalRangeConstraintSet},

	// destruction
	{ 0x8dd02781, 0x30a93a5,  hkVersionRegistry::VERSION_COPY, "hkdBreakableBodyBlueprint", HK_NULL },
	{ 0xa0bd07b9, 0xf72db378, hkVersionRegistry::VERSION_COPY, "hkdSliceFracture", Update_hkdSliceFracture },
	{ 0x91d0dc06, 0x73fb6a85, hkVersionRegistry::VERSION_COPY, "hkdDeformationController", Update_hkdDeformationController },

	{ 0, 0, 0, HK_NULL, HK_NULL }
};

static const hkVersionRegistry::ClassRename s_renames[] =
{
	{ HK_NULL, HK_NULL }
};

#define HK_COMPAT_VERSION_FROM hkHavok600b2Classes
#define HK_COMPAT_VERSION_TO hkHavok600r1Classes
#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO

} // namespace hkCompat_hk600b2_hk600r1

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
