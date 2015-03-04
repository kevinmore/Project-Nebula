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

namespace
{
	struct DummyArray
	{
		void* data;
		int size;
		int capacity;
	};

	struct __NewElem
	{
		__NewElem() { }
		__NewElem(hkUint32 bb,hkUint16 tt, hkUint16 uu ) : b(bb), t(tt), u(uu) { }
		hkUint32 b;
		hkUint16 t;
		hkUint16 u;
	};
}

namespace hkCompat_hk550r1_hk600b1
{
#define REMOVED(TYPE) { 0,0, hkVersionRegistry::VERSION_REMOVED, TYPE, HK_NULL }
#define BINARY_IDENTICAL(OLDSIG,NEWSIG,TYPE) { OLDSIG, NEWSIG, hkVersionRegistry::VERSION_MANUAL, TYPE, HK_NULL }

	static void Update_ignore( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
	}

	static void Update_VertexFormat(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		// Stride
		hkClassMemberAccessor oldStride( oldObj, "stride" );
		hkClassMemberAccessor newStride( newObj, "stride" );
		newStride.asUint32() = oldStride.asUint8();

		// Create new style vertex attributes
		hkClassMemberAccessor pos( oldObj, "positionOffset");
		hkClassMemberAccessor norm( oldObj, "normalOffset");
		hkClassMemberAccessor tangent( oldObj, "tangentOffset");
		hkClassMemberAccessor binorm( oldObj, "binormalOffset");
		hkClassMemberAccessor numBones( oldObj, "numBonesPerVertex");
		hkClassMemberAccessor boneIndex( oldObj, "boneIndexOffset");
		hkClassMemberAccessor boneWeight( oldObj, "boneWeightOffset");
		hkClassMemberAccessor numTextures( oldObj, "numTextureChannels");
		hkClassMemberAccessor floatTexture( oldObj, "tFloatCoordOffset");
		hkClassMemberAccessor quantTexture( oldObj, "tQuantizedCoordOffset");
		hkClassMemberAccessor color( oldObj, "colorOffset");

		hkClassMemberAccessor newDescArray(newObj, "decls");

		enum __DataType
		{
			__DT_NONE = 0,
			__DT_UINT8,
			__DT_INT16,
			__DT_UINT32,
			__DT_FLOAT,
			__DT_FLOAT2,
			__DT_FLOAT3,
			__DT_FLOAT4
		};

		enum __DataUsage
		{
			__DU_NONE = 0,
			__DU_POSITION = 1,
			__DU_COLOR = 2,
			__DU_NORMAL = 4,
			__DU_TANGENT = 8,
			__DU_BINORMAL = 16,
			__DU_TEXCOORD = 32,
			__DU_BLENDWEIGHTS = 64,
			__DU_BLENDINDICES = 128
		};

		const hkUint8 NOT_USED = 255;

		DummyArray& decls = *static_cast<DummyArray*>(newDescArray.asRaw());
		decls.capacity = 7 + numTextures.asUint8();
		decls.size = 0;
		decls.data = hkAllocateChunk<__NewElem>(decls.capacity, HK_MEMORY_CLASS_ARRAY);
#define ADD_NEW_ELEM(b, t, u) \
		{ \
			HK_ASSERT(0x039de78b, decls.size < decls.capacity); \
			__NewElem newData(b, t, u); \
			hkString::memCpy(hkAddByteOffset(decls.data, decls.size++*hkSizeOf(__NewElem)), &newData, hkSizeOf(__NewElem)); \
		}

		if (pos.asUint8() < NOT_USED)
			ADD_NEW_ELEM( pos.asUint8(), __DT_FLOAT4, __DU_POSITION );
		if (norm.asUint8() < NOT_USED)
			ADD_NEW_ELEM( norm.asUint8(), __DT_FLOAT4, __DU_NORMAL );
		if (tangent.asUint8() < NOT_USED)
			ADD_NEW_ELEM( tangent.asUint8(), __DT_FLOAT4, __DU_TANGENT );
		if (binorm.asUint8() < NOT_USED)
			ADD_NEW_ELEM( binorm.asUint8(), __DT_FLOAT4, __DU_BINORMAL );

		if (numBones.asUint8() > 0 )
		{
			if (boneIndex.asUint8() < NOT_USED)
				ADD_NEW_ELEM( boneIndex.asUint8(), __DT_UINT8, __DU_BLENDINDICES );
			if (boneWeight.asUint8() < NOT_USED)
				ADD_NEW_ELEM( boneWeight.asUint8(), __DT_UINT8, __DU_BLENDWEIGHTS );
		}

		if (color.asUint8() < NOT_USED)
			ADD_NEW_ELEM( color.asUint8(), __DT_UINT32, __DU_COLOR );

		int toffset = 0;
		for (int nt = 0; nt < numTextures.asUint8(); ++nt)
		{
			if (floatTexture.asUint8() < NOT_USED)
			{
				ADD_NEW_ELEM( floatTexture.asUint8() + toffset, __DT_FLOAT, __DU_TEXCOORD );
				toffset += sizeof(hkReal) * 2;
			}
			else if (quantTexture.asUint8() < NOT_USED)
			{
				ADD_NEW_ELEM( quantTexture.asUint8() + toffset, __DT_INT16, __DU_TEXCOORD );
				toffset += sizeof(hkInt16) * 2;
			}
		}
		tracker.addChunk(decls.data, hkSizeOf(__NewElem)*decls.capacity, HK_MEMORY_CLASS_ARRAY);
		decls.capacity |= hkArray<char>::DONT_DEALLOCATE_FLAG;
	}

	static void Update_hkbMoveBoneTowardTargetModifier(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		// hkbMoveBoneTowardTargetModifier::m_eventToSendWhenTargetReached
		{
			hkClassMemberAccessor oldEventToSendWhenTargetReached(oldObj, "eventToSendWhenTargetReached");
			hkClassMemberAccessor newEventToSendWhenTargetReached(newObj, "eventToSendWhenTargetReached");

			newEventToSendWhenTargetReached.asInt32() = oldEventToSendWhenTargetReached.asInt32(); // copy hkbEvent::m_id (first member)
		}
	}

	static void Update_hkbTargetRigidBodyModifier(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		// Instead on one sensorBoneIndex and an additional bool, there are now two bone indices, only one of which should be valid.
		{
			hkClassMemberAccessor oldSensorBoneIndex(oldObj, "sensorBoneIndex");
			hkClassMemberAccessor oldSenseFromRagdollBone(oldObj, "senseFromRagdollBone");
			hkClassMemberAccessor newSensorRagdollBoneIndex(newObj, "sensorRagdollBoneIndex");
			hkClassMemberAccessor newSensorAnimationBoneIndex(newObj, "sensorAnimationBoneIndex");

			hkInt16 oldBoneIndex = static_cast<hkInt16>(oldSensorBoneIndex.asInt32());

			if ( oldSenseFromRagdollBone.asBool() )
			{
				newSensorRagdollBoneIndex.asInt16() = oldBoneIndex;
			}
			else
			{
				newSensorAnimationBoneIndex.asInt16() = oldBoneIndex;
			}
		}

		// hkbEvents are now hkInt32s
		{
			hkClassMemberAccessor oldEventToSend(oldObj, "eventToSend");
			hkClassMemberAccessor newEventToSend(newObj, "eventToSend");

			newEventToSend.asInt32() = oldEventToSend.asInt32(); // copy hkbEvent::m_id (first member)
		}
		{
			hkClassMemberAccessor oldEventToSendToTarget(oldObj, "eventToSendToTarget");
			hkClassMemberAccessor newEventToSendToTarget(newObj, "eventToSendToTarget");

			newEventToSendToTarget.asInt32() = oldEventToSendToTarget.asInt32(); // copy hkbEvent::m_id (first member)
		}
	}

	static void Update_hkbAttachmentSetup(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		// hkbAttachmentSetup::m_attacherBoneIndex changed from hkInt32 to hkInt16
		{
			hkClassMemberAccessor oldAttacherBoneIndex(oldObj, "attacherBoneIndex");
			hkClassMemberAccessor newAttacherBoneIndex(newObj, "attacherBoneIndex");

			newAttacherBoneIndex.asInt16() = static_cast<hkInt16>(oldAttacherBoneIndex.asInt32());
		}
		// hkbAttachmentSetup::m_attacheeBoneIndex changed from hkInt32 to hkInt16
		{
			hkClassMemberAccessor oldAttacheeBoneIndex(oldObj, "attacheeBoneIndex");
			hkClassMemberAccessor newAttacheeBoneIndex(newObj, "attacheeBoneIndex");

			newAttacheeBoneIndex.asInt16() = static_cast<hkInt16>(oldAttacheeBoneIndex.asInt32());
		}
	}

	static void Update_hkbConstrainRigidBodyModifier(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		{
			hkClassMemberAccessor oldConstraintType(oldObj, "constraintType");
			hkClassMemberAccessor newConstraintType(newObj, "constraintType");

			// TYPE_ENUM.TYPE_INT8 = TYPE_INT8.TYPE_VOID
			newConstraintType.asInt8() = oldConstraintType.asInt8();
		}

		// changed from int to hkInt16
		{
			hkClassMemberAccessor oldRagdollBoneToConstrain(oldObj, "ragdollBoneToConstrain");
			hkClassMemberAccessor newRagdollBoneToConstrain(newObj, "ragdollBoneToConstrain");

			newRagdollBoneToConstrain.asInt16() = static_cast<hkInt16>(oldRagdollBoneToConstrain.asInt32());
		}
	}

	static void Update_hkbAttachmentModifierAttachmentProperties(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor oldSensingRagdollBoneIndex(oldObj, "sensingRagdollBoneIndex");
		hkClassMemberAccessor newSensingRagdollBoneIndex(newObj, "sensingRagdollBoneIndex");

		newSensingRagdollBoneIndex.asInt16() = static_cast<hkInt16>(oldSensingRagdollBoneIndex.asInt32());
	}

	static void Update_hkbCheckRagdollSpeedModifier(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor oldEventToSend(oldObj, "eventToSend");
		hkClassMemberAccessor newEventToSend(newObj, "eventToSend");

		newEventToSend.asInt32() = oldEventToSend.asInt32(); // copy hkbEvent::m_id (first member)
	}

	static void Update_hkbAttachmentModifier(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor oldAttachmentProperties(oldObj, "attachmentProperties");
		hkClassMemberAccessor newAttachmentProperties(newObj, "attachmentProperties");

		// hkbAttachmentModifier::m_attachmentProperties
		hkClassMemberAccessor::SimpleArray& oldArray = oldAttachmentProperties.asSimpleArray();
		hkClassMemberAccessor::SimpleArray& newArray = newAttachmentProperties.asSimpleArray();
		HK_ASSERT(0xad785560, oldArray.size == newArray.size);
		const hkClass* oldClass = oldAttachmentProperties.getClassMember().getClass();
		const hkClass* newClass = newAttachmentProperties.getClassMember().getClass();
		for(int i = 0; i < oldArray.size; ++i )
		{
			hkVariant oldObjItem = {static_cast<char*>(oldArray.data) + oldClass->getObjectSize()*i, oldClass};
			hkVariant newObjItem = {static_cast<char*>(newArray.data) + newClass->getObjectSize()*i, newClass};

			Update_hkbAttachmentModifierAttachmentProperties(oldObjItem, newObjItem, tracker);
		}
	}

	static void Update_hkbReachTowardTargetModifier(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		// the hand data is now in an array of hands instead of two arrays
		hkClassMemberAccessor newLeftHand(newObj, "leftHand");
		hkClassMemberAccessor newRightHand(newObj, "rightHand");
		hkClassMemberAccessor oldHandIndex(oldObj, "handIndex");
		hkClassMemberAccessor oldIsHandEnabled(oldObj, "isHandEnabled");

		// left hand
		{
			hkVariant newLeftHandVariant;
			newLeftHandVariant.m_class = newLeftHand.getClassMember().getClass();
			newLeftHandVariant.m_object = newLeftHand.getAddress();

			hkClassMemberAccessor newHandIndex(newLeftHandVariant, "handIndex");
			newHandIndex.asInt16() = oldHandIndex.asInt16(0);

			hkClassMemberAccessor newIsHandEnabled(newLeftHandVariant, "isHandEnabled" );
			newIsHandEnabled.asBool() = oldIsHandEnabled.asBool(0);
		}

		// right hand
		{
			hkVariant newRightHandVariant;
			newRightHandVariant.m_class = newRightHand.getClassMember().getClass();
			newRightHandVariant.m_object = newRightHand.getAddress();

			hkClassMemberAccessor newHandIndex(newRightHandVariant, "handIndex");
			newHandIndex.asInt16() = oldHandIndex.asInt16(1);

			hkClassMemberAccessor newIsHandEnabled(newRightHandVariant, "isHandEnabled" );
			newIsHandEnabled.asBool() = oldIsHandEnabled.asBool(1);
		}
	}

static const hkVersionRegistry::ClassAction s_updateActions[] =
{
	// common
	{ 0x8bdd3e9a, 0x6728e4b7, hkVersionRegistry::VERSION_VARIANT | hkVersionRegistry::VERSION_COPY, "hkaBoneAttachment", HK_NULL },
	{ 0xf598a34e, 0xf598a34e, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainer", HK_NULL },
	{ 0x853a899c, 0x853a899c, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainerNamedVariant", HK_NULL },
	{ 0xf2ec0c9c, 0xBE6765DD, hkVersionRegistry::VERSION_VARIANT | hkVersionRegistry::VERSION_COPY, "hkxMaterial", Update_ignore },
	{ 0x914da6c1, 0x914da6c1, hkVersionRegistry::VERSION_VARIANT, "hkxAttribute", HK_NULL },
	{ 0x1667c01c, 0x1667c01c, hkVersionRegistry::VERSION_VARIANT, "hkxAttributeGroup", HK_NULL },
	{ 0x06af1b5a, 0x06af1b5a, hkVersionRegistry::VERSION_VARIANT, "hkxNode", HK_NULL },
	{ 0x72e8e849, 0x72e8e849, hkVersionRegistry::VERSION_VARIANT, "hkxMesh", HK_NULL },
	{ 0x912c8863, 0x912c8863, hkVersionRegistry::VERSION_VARIANT, "hkxMeshSection", HK_NULL },
	{ 0x64e9a03c, 0x64e9a03c, hkVersionRegistry::VERSION_VARIANT, "hkxMeshUserChannelInfo", HK_NULL },
	{ 0x445a443a, 0x445a443a, hkVersionRegistry::VERSION_VARIANT, "hkxAttributeHolder", HK_NULL },
	{ 0x1388d601, 0x1388d601, hkVersionRegistry::VERSION_VARIANT, "hkCustomAttributesAttribute", HK_NULL },
	{ 0xbff19005, 0xbff19005, hkVersionRegistry::VERSION_VARIANT, "hkCustomAttributes", HK_NULL },
	{ 0xe085ba9f, 0x02ea23f0, hkVersionRegistry::VERSION_VARIANT | hkVersionRegistry::VERSION_COPY, "hkxMaterialTextureStage", Update_ignore }, // TODO : enum values
	{ 0xe30984ec, 0xf6a7f6f9, hkVersionRegistry::VERSION_MANUAL, "hkxMaterialEffect", Update_ignore },
	{ 0x57061454, 0x8e9b1727, hkVersionRegistry::VERSION_HOMOGENEOUSARRAY, "hkxVertexBuffer", HK_NULL },

	// hkaAnnotationTrack
	BINARY_IDENTICAL(0x846fc690, 0xba632fd5, "hkaAnnotationTrack"), // name -> trackName

	// hkscenedata
	{ 0x379fd194, 0xe745584c, hkVersionRegistry::VERSION_COPY, "hkxVertexFormat", Update_VertexFormat },
	{ 0x035deb8a, 0x74c3397a, hkVersionRegistry::VERSION_MANUAL, "hkxVertexP4N4C1T2", Update_ignore },
	{ 0xb99f5900, 0xce018bf0, hkVersionRegistry::VERSION_MANUAL, "hkxVertexP4N4T4B4C1T2", Update_ignore },

	// physics
	{ 0x8c608221, 0x58e1e585, hkVersionRegistry::VERSION_COPY, "hkpTriSampledHeightFieldBvTreeShape", HK_NULL },
	{ 0xe7eca7eb, 0xa823d623, hkVersionRegistry::VERSION_COPY, "hkpBvTreeShape", HK_NULL },
	{ 0xfa798537, 0xc84eafb1, hkVersionRegistry::VERSION_COPY, "hkpEntity", HK_NULL },	
	{ 0xad62b3cc, 0x11213421, hkVersionRegistry::VERSION_COPY, "hkpSampledHeightFieldShape", HK_NULL }, // m_heightfieldType will get fixed up in finishLoading constructor

	{ 0xa4cbb846, 0xcda551fc, hkVersionRegistry::VERSION_COPY, "hkpRagdollMotorConstraintAtom", HK_NULL },
	{ 0xeb0cd053, 0x31f375e2, hkVersionRegistry::VERSION_COPY, "hkpRagdollConstraintData", HK_NULL },
	{ 0xbcca03b0, 0xbb1e4ebc, hkVersionRegistry::VERSION_COPY, "hkpRagdollConstraintDataAtoms", HK_NULL },

	BINARY_IDENTICAL(0x97b45527, 0xa8035513, "hkpCollidableBoundingVolumeData"),
	BINARY_IDENTICAL(0x5879a2c3, 0x19e24f2b, "hkpCollidable"),
	BINARY_IDENTICAL(0x50f6ee9f, 0xcebb2443, "hkpWorldObject"),

	BINARY_IDENTICAL(0x28c7f0f3, 0xfb925fd5, "hkpSerializedAgentNnEntry"),

	// collide
	{  0x260c3908, 0x428d3fd6, hkVersionRegistry::VERSION_COPY, "hkpExtendedMeshShape", HK_NULL },
	{  0xfce016a3, 0xd42fac97, hkVersionRegistry::VERSION_COPY, "hkpExtendedMeshShapeShapesSubpart", HK_NULL },

	// behavior
	REMOVED("hkbAdditiveBinaryBlenderGenerator"),
	REMOVED("hkbBinaryBlenderGenerator"),
	REMOVED("hkbVariableSetVariable"),
	REMOVED("hkbVariableSet"),
	REMOVED("hkbVariableSetTarget"),
	{ 0x6ada7bd9, 0xed3ea576, hkVersionRegistry::VERSION_COPY, "hkbRigidBodyRagdollModifier", HK_NULL }, // added m_keyframedBonesList, removed m_keyframedBones, HKF-586
	{ 0x2a3617a5, 0x9e27ca98, hkVersionRegistry::VERSION_COPY, "hkbMoveBoneTowardTargetModifier", Update_hkbMoveBoneTowardTargetModifier }, // changed/added members, HKF-586
	{ 0xcc4f7446, 0x64333cca, hkVersionRegistry::VERSION_COPY, "hkbTargetRigidBodyModifier", Update_hkbTargetRigidBodyModifier }, // changed enums, added/removed members, HKF-356, HKF-588, HKF-612
	{ 0xa6004a7c, 0xb3222567, hkVersionRegistry::VERSION_COPY, "hkbPoweredRagdollModifier", HK_NULL }, // added members, HKF-586
	{ 0x4cce6ed9, 0x86028e60, hkVersionRegistry::VERSION_COPY, "hkbAttachmentSetup", Update_hkbAttachmentSetup }, // changed members, HKF-591, HKF-588
	{ 0x78be807b, 0xfd32f68b, hkVersionRegistry::VERSION_COPY, "hkbConstrainRigidBodyModifier", Update_hkbConstrainRigidBodyModifier }, // added m_variableTarget and m_clearTargetData, changed m_constraintType, HKF-586
	{ 0xbbbd0506, 0xa8a34bcc, hkVersionRegistry::VERSION_COPY, "hkbPositionRelativeSelectorGenerator", HK_NULL }, // added m_fixPositionEventId, HKF-521
	{ 0x2cf42b86, 0x257691a0, hkVersionRegistry::VERSION_COPY, "hkbAttachmentModifier", Update_hkbAttachmentModifier }, // added m_attachmentSetup, changes in type of m_attachmentProperties, HKF-591
	{ 0xda11d903, 0x362a479c, hkVersionRegistry::VERSION_COPY, "hkbAttachmentModifierAttachmentProperties", Update_hkbAttachmentModifierAttachmentProperties }, // changed m_sensingRagdollBoneIndex, HKF-591
	{ 0xc5b3a056, 0x60efa34c, hkVersionRegistry::VERSION_COPY, "hkbReachModifier", HK_NULL }, // added members, HKF-587
	{ 0x6db151b0, 0x2d6a1e8a, hkVersionRegistry::VERSION_COPY, "hkbDelayedModifier", HK_NULL }, // moved members, HKF-612
	{ 0x8b11c9e6, 0xd16cbb5f, hkVersionRegistry::VERSION_COPY, "hkbFaceTargetModifier", HK_NULL }, // added m_target, HKF-612
	{ 0x61881b16, 0x44fe7427, hkVersionRegistry::VERSION_COPY, "hkbBlenderGenerator", HK_NULL }, // changed FlagBits enums, HKD-9
	{ 0x5b49cdd5, 0x7321fd67, hkVersionRegistry::VERSION_COPY, "hkbBlendingTransitionEffect", HK_NULL }, // m_childFrequencies +nosave, HKF-414
	{ 0x66ba5c35, 0x407d20dc, hkVersionRegistry::VERSION_COPY, "hkbCheckRagdollSpeedModifier", Update_hkbCheckRagdollSpeedModifier }, // changed type of m_eventToSend, HKF-612
	{ 0x1551f22d, 0x63a20ff7, hkVersionRegistry::VERSION_COPY, "hkbClipGenerator", HK_NULL }, // changes in type of m_triggers, changed Flags enums, HKD-9
	{ 0xb0ede491, 0xbaa06e0b, hkVersionRegistry::VERSION_COPY, "hkVariableTweakingHelper", HK_NULL },
	{ 0xcd849481, 0xe49e625e, hkVersionRegistry::VERSION_COPY, "hkbFootIkControlData", HK_NULL },
	{ 0xbf1ce00e, 0x713555b8, hkVersionRegistry::VERSION_COPY, "hkbFootIkControlsModifier", HK_NULL },
	{ 0xf5264fd4, 0x8a61ce23, hkVersionRegistry::VERSION_COPY, "hkbGeneratorTransitionEffect", HK_NULL },
	{ 0xb707333a, 0x4a6c28da, hkVersionRegistry::VERSION_COPY, "hkbLookAtModifier", HK_NULL },
	{ 0x6b99270b, 0x2f0cf6c5, hkVersionRegistry::VERSION_COPY, "hkbBehaviorGraphData", HK_NULL },
	{ 0x1baa90d7, 0xf927a830, hkVersionRegistry::VERSION_COPY, "hkbCharacter", HK_NULL },
	{ 0x622ff0bf, 0x2d995d9c, hkVersionRegistry::VERSION_COPY, "hkbTarget", HK_NULL },
	{ 0xa3049f36, 0x931b5a33, hkVersionRegistry::VERSION_COPY, "hkbFootIkModifier", HK_NULL },
	{ 0xe9bbd108, 0xd734aed8, hkVersionRegistry::VERSION_COPY, "hkbStateMachine", HK_NULL },
	{ 0x87c3c3aa, 0x2b5fb060, hkVersionRegistry::VERSION_COPY, "hkbReachTowardTargetModifier", Update_hkbReachTowardTargetModifier },
	{ 0x86fa3de7, 0x1e9bec06, hkVersionRegistry::VERSION_COPY, "hkbBehaviorGraph", HK_NULL },
	{ 0xc376dfcc, 0x0160d338, hkVersionRegistry::VERSION_COPY, "hkbNode", HK_NULL },
	{ 0x9af27949, 0x4f6a5aec, hkVersionRegistry::VERSION_COPY, "hkbFootIkGains", HK_NULL },
	BINARY_IDENTICAL(0x306813b4, 0xcf63a99, "hkbVariableInfo"),
	
	{ 0xcdb31e0c, 0x4eae6610, hkVersionRegistry::VERSION_COPY, "hkaMeshBinding", HK_NULL },
	{ 0xfb496074, 0xbd7f7a93, hkVersionRegistry::VERSION_COPY, "hkaAnimationBinding", HK_NULL },

	{ 0, 0, 0, HK_NULL, HK_NULL }
};

static const hkVersionRegistry::ClassRename s_renames[] =
{
	{ "hkxVertexFormat", "hkxVertexDescription" },
	{ "hkbModifierSequence", "hkbModifierList" },

	{ "hkaSkeletalAnimation", "hkaAnimation" },
	{ "hkaInterleavedSkeletalAnimation", "hkaInterleavedUncompressedAnimation" },
	{ "hkaDeltaCompressedSkeletalAnimation", "hkaDeltaCompressedAnimation" },
	{ "hkaWaveletSkeletalAnimation", "hkaWaveletCompressedAnimation" },
	//{ "hkaMirroredSkeletalAnimation", "hkaMirroredAnimation" }, // not reflected
	{ "hkaSplineSkeletalAnimation", "hkaSplineCompressedAnimation" },

	{ "hkaSplineSkeletalAnimationTrackCompressionParams", "hkaSplineCompressedAnimationTrackCompressionParams" },
	{ "hkaSplineSkeletalAnimationAnimationCompressionParams", "hkaSplineCompressedAnimationAnimationCompressionParams" },

	{ "hkaDeltaCompressedSkeletalAnimationQuantizationFormat", "hkaDeltaCompressedAnimationQuantizationFormat" },
	{ "hkaWaveletSkeletalAnimationCompressionParams", "hkaWaveletCompressedAnimationCompressionParams" },
	{ "hkaWaveletSkeletalAnimationQuantizationFormat", "hkaWaveletCompressedAnimationQuantizationFormat" },

	{ HK_NULL, HK_NULL }
};

#define HK_COMPAT_VERSION_FROM hkHavok550r1Classes
#define HK_COMPAT_VERSION_TO hkHavok600b1Classes
#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO

} // namespace hkCompat_hk550r1_hk600b1

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
