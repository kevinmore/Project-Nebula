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

namespace hkCompat_hk600b1_hk600b2
{
#define REMOVED(TYPE) { 0,0, hkVersionRegistry::VERSION_REMOVED, TYPE, HK_NULL }
#define BINARY_IDENTICAL(OLDSIG,NEWSIG,TYPE) { OLDSIG, NEWSIG, hkVersionRegistry::VERSION_MANUAL, TYPE, HK_NULL }

	static void Update_ignore( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
	}

	static void Update_hkBitField( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldWords(oldObj, "words"); // TYPE_SIMPLEARRAY/TYPE_UINT32
		hkClassMemberAccessor oldNumBitsAndFlags(oldObj, "numBitsAndFlags"); // TYPE_INT32
		hkClassMemberAccessor newWords(newObj, "words"); // TYPE_ARRAY/TYPE_UINT32
		hkClassMemberAccessor newNumBits(newObj, "numBits"); // TYPE_INT32

		struct BitFieldArray
		{
			void* data;
			hkInt32 size;
			hkInt32 capacity;
		};
		hkClassMemberAccessor::SimpleArray& oldWordsArray = oldWords.asSimpleArray();
		BitFieldArray& newWordsArray = *static_cast<BitFieldArray*>(newWords.asRaw());

		// set words
		newWordsArray.capacity = newWordsArray.size = oldWordsArray.size;
		newWordsArray.data = oldWordsArray.data;
		newWordsArray.capacity |= hkArray<char>::DONT_DEALLOCATE_FLAG;
		// set num of bits
		newNumBits.asInt32() = oldNumBitsAndFlags.asInt32() & 0x7fffffff;
	}

	static void Update_hkbRigidBodyRagdollModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldKeyframedBones(oldObj, "keyframedBones"); // hkBitField
		hkVariant oldBitField = { oldKeyframedBones.object().getAddress(), &oldKeyframedBones.object().getClass() };
		hkClassMemberAccessor newKeyframedBones(newObj, "keyframedBones"); // hkBitField
		hkVariant newBitField = { newKeyframedBones.object().getAddress(), &newKeyframedBones.object().getClass() };

		Update_hkBitField( oldBitField, newBitField, tracker );
	}

	static void Update_hkbPoweredRagdollModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldKeyframedBones(oldObj, "keyframedBones"); // hkBitField
		hkVariant oldBitField = { oldKeyframedBones.object().getAddress(), &oldKeyframedBones.object().getClass() };
		hkClassMemberAccessor newKeyframedBones(newObj, "keyframedBones"); // hkBitField
		hkVariant newBitField = { newKeyframedBones.object().getAddress(), &newKeyframedBones.object().getClass() };

		Update_hkBitField( oldBitField, newBitField, tracker );
	}

	static void Update_hkbKeyframeBonesModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldKeyframedBones(oldObj, "keyframedBones"); // hkBitField
		hkVariant oldBitField = { oldKeyframedBones.object().getAddress(), &oldKeyframedBones.object().getClass() };
		hkClassMemberAccessor newKeyframedBones(newObj, "keyframedBones"); // hkBitField
		hkVariant newBitField = { newKeyframedBones.object().getAddress(), &newKeyframedBones.object().getClass() };

		Update_hkBitField( oldBitField, newBitField, tracker );
	}

	static void Update_hkbPositionRelativeSelectorGenerator( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldAxis(oldObj, "targetRotationAxis");
		hkClassMemberAccessor oldAngle(oldObj, "targetRotationAngle");
		hkClassMemberAccessor newRotation(newObj, "targetRotation");

		// get the axis as an hkVector4
		hkVector4 axis;
		axis(0) = oldAxis.asVector4().r[0];
		axis(1) = oldAxis.asVector4().r[1];
		axis(2) = oldAxis.asVector4().r[2];
		axis(3) = oldAxis.asVector4().r[3];

		// compute a quaternion from the old (axis,angle)
		hkQuaternion q( axis, oldAngle.asReal() );

		// store the quaternion in the new member
		newRotation.asVector4().r[0] = q.m_vec(0);
		newRotation.asVector4().r[1] = q.m_vec(1);
		newRotation.asVector4().r[2] = q.m_vec(2);
		newRotation.asVector4().r[3] = q.m_vec(3);
	}

	static void Update_hkpTriSampledHeightFieldCollection( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor newExtrusion(newObj, "triangleExtrusion");

		hkClassMemberAccessor::Vector4& extVec = newExtrusion.asVector4();
		extVec.r[0] = 0.0f;
		extVec.r[1] = 0.0f;
		extVec.r[2] = 0.0f;
		extVec.r[3] = 0.0f;
	}

	union fmtU16
	{
		hkUint8 b[2];
		hkUint16 v;
	};

	static inline hkUint16 hkConvertEndianU16( hkUint16 n )
	{
		union fmtU16 fDataIn, fDataOut;
		fDataIn.v = n;
		fDataOut.b[0] = fDataIn.b[1];
		fDataOut.b[1] = fDataIn.b[0];
		return fDataOut.v;
	}

	static void hkaTrackAnalysis_calcStats( const hkUint16* staticMaskT, hkUint32 numTracksT, hkBool endianFlip, int& staticDOFsOut, int& dynamicDOFsOut )
	{
		enum TrackType
		{
			HK_TRACK_DYNAMIC = 0x00, // This track should use the dynamic value
			HK_TRACK_STATIC	 = 0x01, // This track should use the static value
			HK_TRACK_CLEAR   = 0x02	 // This track can be cleared / set to the identity
		};

		int totalDOFs = numTracksT * 10;

		// Clear DOFs
		int clearDOFs = 0;
		for (hkUint32 i=0; i < numTracksT; i++)
		{
			hkUint16 staticMaskTi = staticMaskT[i];
			if(endianFlip)
			{
				staticMaskTi = hkConvertEndianU16(staticMaskTi);
			}

			if ( (staticMaskTi & 0x3) == HK_TRACK_CLEAR )
			{
				// Position is clear
				clearDOFs+=3;
			}
			if ( ((staticMaskTi >> 2) & 0x3) == HK_TRACK_CLEAR )
			{
				// Rotation is clear
				clearDOFs+=4;
			}
			if ( ((staticMaskTi >> 4) & 0x3) == HK_TRACK_CLEAR )
			{
				// Scale is clear
				clearDOFs+=3;
			}

		}

		// Dynamic DOFs
		int dynamicDOFs = 0;
		for (hkUint32 j=0; j < numTracksT; j++)
		{
			hkUint16 staticMaskTj = staticMaskT[j];
			if(endianFlip)
			{
				staticMaskTj = hkConvertEndianU16(staticMaskTj);
			}

			hkUint32 detail = (staticMaskTj >> 6);
			// Check each DOF
			for (int d=0; d < 10; d++)
			{
				dynamicDOFs += (detail >> d) & 0x1;
			}
		}

		dynamicDOFsOut = dynamicDOFs;
		staticDOFsOut = totalDOFs - dynamicDOFs - clearDOFs;
	}

	static void Update_hkaWaveletOrDeltaCompressedAnimation(  hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{

		hkClassMemberAccessor oldNumTransformTracks(oldObj, "numberOfTransformTracks"); // TYPE_UINT32
		hkUint32 numTransformTracks = oldNumTransformTracks.asInt32();

		hkClassMemberAccessor oldStaticMaskIdx(oldObj, "staticMaskIdx"); // TYPE_UINT32
		hkUint32 staticMaskIdx = oldStaticMaskIdx.asInt32();

		hkClassMemberAccessor oldDataBuffer(oldObj, "dataBuffer"); // TYPE_UINT8*
		const hkUint8* dataBuffer = reinterpret_cast<const hkUint8*> (oldDataBuffer.asPointer());

		const hkUint16* staticMaskPtr = reinterpret_cast<const hkUint16*>(dataBuffer + staticMaskIdx);

		hkBool doEndianFlip = false;
		if(staticMaskIdx == 4)
		{
			// This is a 'new' animation with the first byte indicating endianness
			doEndianFlip = (dataBuffer[0] != HK_ENDIAN_BIG);
		}
		else
		{
			// This is an 'old' animation where there is no endianness indication. So assume it is LITTLE
			doEndianFlip = (HK_ENDIAN_BIG == 1);
		}

		hkClassMemberAccessor newNumStaticTransformDOFs(newObj, "numStaticTransformDOFs"); // TYPE_UINT32
		hkClassMemberAccessor newNumDynamicTransformDOFs(newObj, "numDynamicTransformDOFs"); // TYPE_UINT32

		int staticTransformDOFs, dynamicTransformDOFs;
		hkaTrackAnalysis_calcStats( staticMaskPtr, numTransformTracks, doEndianFlip, staticTransformDOFs, dynamicTransformDOFs );
		newNumStaticTransformDOFs.asInt32() = staticTransformDOFs;
		newNumDynamicTransformDOFs.asInt32() = dynamicTransformDOFs;
	}


static const hkVersionRegistry::ClassAction s_updateActions[] =
{
	// common
	{ 0x6728e4b7, 0x6728e4b7, hkVersionRegistry::VERSION_VARIANT, "hkaBoneAttachment", HK_NULL },
	{ 0xf598a34e, 0xf598a34e, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainer", HK_NULL },
	{ 0x853a899c, 0x853a899c, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainerNamedVariant", HK_NULL },
	{ 0xbe6765dd, 0xbe6765dd, hkVersionRegistry::VERSION_VARIANT, "hkxMaterial", Update_ignore },
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

	{ 0xbff7593a, 0x49557cc2, hkVersionRegistry::VERSION_COPY, "hkxVertexFloatDataChannel", HK_NULL },

	// base
	{ 0xf2ed2387, 0xda41bd9b, hkVersionRegistry::VERSION_COPY, "hkBitField", Update_hkBitField },

	// physics
	{ 0xc84eafb1, 0xe2f64d5c, hkVersionRegistry::VERSION_COPY, "hkpEntity", HK_NULL },
	{ 0xcb2ecf39, 0x4788428f, hkVersionRegistry::VERSION_COPY, "hkpTriSampledHeightFieldCollection", Update_hkpTriSampledHeightFieldCollection },
	{ 0x9b1a3265, 0x208eee42, hkVersionRegistry::VERSION_COPY, "hkpShapeCollection", HK_NULL },
	{ 0x0b34e763, 0x787ef513, hkVersionRegistry::VERSION_COPY, "hkpTransformShape", HK_NULL }, // added new member m_childShapeSize, HVK-4604
	BINARY_IDENTICAL(0x5b8bc234, 0x5ba0a5f7, "hkpConvexTranslateShape"), // some members moved into the new base class, HVK-4604
	BINARY_IDENTICAL(0x99da124b, 0xae3e5017, "hkpConvexTransformShape"), // some members moved into the new base class, HVK-4604 
	
	// behavior
	BINARY_IDENTICAL( 0xe9feed05, 0x1b0d2fac, "hkbHandIkModifierHand" ),
	{ 0x7321fd67, 0xbc78b8cf, hkVersionRegistry::VERSION_MANUAL, "hkbBlendingTransitionEffect", Update_ignore }, // m_flags changed type
	{ 0xd74c0091, 0xf2c6af3f, hkVersionRegistry::VERSION_COPY, "hkbCustomTestGenerator", HK_NULL },
	{ 0x2d6a1e8a, 0xd51f8ab3, hkVersionRegistry::VERSION_COPY, "hkbDelayedModifier", HK_NULL },
	{ 0x931b5a33, 0x78a6ee24, hkVersionRegistry::VERSION_COPY, "hkbFootIkModifier", HK_NULL },
	{ 0x29bd3613, 0x8c68ad11, hkVersionRegistry::VERSION_COPY, "hkbGeneratorOutput", HK_NULL },
	{ 0x75590584, 0xe08cfea7, hkVersionRegistry::VERSION_COPY, "hkbHandIkControlData", HK_NULL },
	{ 0xf6c1dd9c, 0xf5fb3115, hkVersionRegistry::VERSION_COPY, "hkbHandIkModifier", HK_NULL },
	{ 0xc3354b07, 0xd6e80702, hkVersionRegistry::VERSION_COPY, "hkbKeyframeBonesModifier", Update_hkbKeyframeBonesModifier },
	{ 0x3321bee8, 0xf40f4e8a, hkVersionRegistry::VERSION_COPY, "hkbMoveCharacterModifier", HK_NULL },
	{ 0xa8a34bcc, 0x87579894, hkVersionRegistry::VERSION_COPY, "hkbPositionRelativeSelectorGenerator", Update_hkbPositionRelativeSelectorGenerator },
	{ 0xb3222567, 0x9f4d8c93, hkVersionRegistry::VERSION_COPY, "hkbPoweredRagdollModifier", Update_hkbPoweredRagdollModifier },
	{ 0xed3ea576, 0xdb2aa071, hkVersionRegistry::VERSION_COPY, "hkbRigidBodyRagdollModifier", Update_hkbRigidBodyRagdollModifier },
	{ 0xc12a6e56, 0x9768b7c3, hkVersionRegistry::VERSION_COPY, "hkbSplinePathGenerator", HK_NULL },
	BINARY_IDENTICAL(0x63a20ff7, 0xa4b5bbc1, "hkbClipGenerator"), // renamed enum Flags to ClipFlags
	BINARY_IDENTICAL(0xac31d210, 0x0718b7a9, "hkbStateMachineTransitionInfo"), // renamed enum FlagBits to TransitionFlags
	BINARY_IDENTICAL(0xe2b1c2f3, 0x0f7b063e, "hkbStateMachineStateInfo"), // changes in hkbStateMachineTransitionInfo
	BINARY_IDENTICAL(0x44fe7427, 0x06dc4ef7, "hkbBlenderGenerator"), // renamed enum FlagBits to BlenderFlags
	{ 0xd734aed8, 0xabacb11c, hkVersionRegistry::VERSION_COPY, "hkbStateMachine", HK_NULL },

	// animation
	{ 0x5974ffe8, 0xd1993818, hkVersionRegistry::VERSION_COPY, "hkaWaveletCompressedAnimation", Update_hkaWaveletOrDeltaCompressedAnimation },
	{ 0x83c6794c, 0xf0b3f7d1, hkVersionRegistry::VERSION_COPY, "hkaDeltaCompressedAnimation", Update_hkaWaveletOrDeltaCompressedAnimation },
	
	{ 0, 0, 0, HK_NULL, HK_NULL }
};

static const hkVersionRegistry::ClassRename s_renames[] =
{
	{ HK_NULL, HK_NULL }
};

#define HK_COMPAT_VERSION_FROM hkHavok600b1Classes
#define HK_COMPAT_VERSION_TO hkHavok600b2Classes
#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO

} // namespace hkCompat_hk600b1_hk600b2

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
