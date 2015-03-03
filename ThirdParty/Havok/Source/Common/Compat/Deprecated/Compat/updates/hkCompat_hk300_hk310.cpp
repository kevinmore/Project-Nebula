/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Compat/Deprecated/Compat/hkCompatUtil.h>
#include <Common/Base/Container/StringMap/hkStringMap.h>
#include <Common/Base/Math/hkMath.h>

#include <Common/Serialize/Copier/hkObjectCopier.h>
#include <Common/Serialize/Version/hkVersionUtil.h>
#include <Common/Compat/Deprecated/Version/hkVersionRegistry.h>
#include <Common/Serialize/Util/hkStaticClassNameRegistry.h>

#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>

namespace hkCompat_hk300_hk310
{

	namespace
	{
		template <typename T>
		struct SimpleArray
		{
			T* data;
			int size;
		};
	}

	//
	// Physics
	//

	static void HK_CALL RagdollConstraintData( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& )
	{
		hkClassMemberAccessor oldAnglesAreNotStoredInHavok30Format(oldObj, "anglesAreNotStoredInHavok30Format");
		if( oldAnglesAreNotStoredInHavok30Format.isOk() && oldAnglesAreNotStoredInHavok30Format.asBool() == false )
		{
			HK_ASSERT( 0x5bf9f734, oldObj.m_object == newObj.m_object );
			hkClassMemberAccessor coneMinAngle(oldObj, "coneMinAngle");
			hkClassMemberAccessor planeMaxAngle(oldObj, "planeMaxAngle");
			hkClassMemberAccessor planeMinAngle(oldObj, "planeMinAngle");
			if( coneMinAngle.isOk() && planeMaxAngle.isOk() && planeMinAngle.isOk() )
			{
				oldAnglesAreNotStoredInHavok30Format.asBool() = true;

				coneMinAngle.asReal() = hkMath::acos(coneMinAngle.asReal());
				planeMaxAngle.asReal() = hkMath::asin(planeMaxAngle.asReal());
				planeMinAngle.asReal() = hkMath::asin(planeMinAngle.asReal());
			}
		}
	}

	static void HK_CALL RagdollInstance( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& )
	{
		hkClassMemberAccessor oldPhysicsSystem(oldObj, "physicsSystem");
		hkClassMemberAccessor newRigidBodies(newObj, "rigidBodies");
		hkClassMemberAccessor newConstraints(newObj, "constraints");
		if( oldPhysicsSystem.isOk() && newRigidBodies.isOk() && newConstraints.isOk() )
		{
			const hkClass& oldPhysicsSystemClass = oldPhysicsSystem.getClassMember().getStructClass();
			hkClassMemberAccessor oldRigidBodies( oldObj.m_object, oldPhysicsSystemClass, "rigidBodies");
			hkClassMemberAccessor oldConstraints( oldObj.m_object, oldPhysicsSystemClass, "constraints");
			newRigidBodies.asSimpleArray() = oldRigidBodies.asSimpleArray();
			newConstraints.asSimpleArray() = oldConstraints.asSimpleArray();
		}
	}

	static void HK_CALL WorldCinfo( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& )
	{
		hkClassMemberAccessor contactRestingVelocity(newObj.m_object, *newObj.m_class, "contactRestingVelocity" );
		if( contactRestingVelocity.isOk() )
		{
			contactRestingVelocity.asReal() = HK_REAL_MAX;
		}
	}

	//
	// animation
	//

	static void HK_CALL SkeletalAnimation( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& )
	{
		hkVersionUtil::renameMember( oldObj, "period", newObj, "duration");
		hkVersionUtil::renameMember( oldObj, "animationTracks", newObj, "annotationTracks");

		hkClassMemberAccessor oldNumberOfBoneTracks(oldObj, "numberOfBoneTracks");
		hkClassMemberAccessor newNumberOfTracks(newObj, "numberOfTracks");
		if( oldNumberOfBoneTracks.isOk() && newNumberOfTracks.isOk() )
		{
			newNumberOfTracks.asInt32() = oldNumberOfBoneTracks.asInt16();
		}
	}

	static void HK_CALL AnimationBinding(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& )
	{
		HK_ASSERT2(0x0878de76, false, "The AnimationBinding update function is broken and is not expected to be called.");
		hkClassMemberAccessor mapping(oldObj, "mapping");
		hkClassMemberAccessor animationTrackToBoneIndices(newObj, "animationTrackToBoneIndices");
		if( mapping.isOk() && animationTrackToBoneIndices.isOk() )
		{
			hkArray<hkInt16> newArray;
			SimpleArray<hkInt16>& oldArray = *(SimpleArray<hkInt16>*)mapping.asRaw();

			for( int j = 0; j < oldArray.size; ++j )
			{
				for( int i = 0; i < oldArray.size; ++i )
				{
					if( oldArray.data[i] == j )
					{
						newArray.pushBack( hkInt16(i) );
						break;
					}
				}
			}
			hkString::memCpy( oldArray.data, newArray.begin(), newArray.getSize() * sizeof(hkInt16) );
			oldArray.size = newArray.getSize();
		}
	}

	static void HK_CALL Bone(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor oldInitialTransform(oldObj, "initialTransform");
		if( oldInitialTransform.isOk() )
		{
			// zero for debugging
			static_cast<hkQsTransform*>(oldInitialTransform.asRaw())->setZero();
		}

		hkClassMemberAccessor oldLockTranslation(oldObj, "lockTranslation");
		hkClassMemberAccessor newLockTranslation(newObj, "lockTranslation");
		if( oldLockTranslation.isOk() && newLockTranslation.isOk() )
		{
			newLockTranslation.asBool() = oldLockTranslation.asBool();
		}
	}

	static void HK_CALL BoneAttachment( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& )
	{
		hkVersionUtil::renameMember( oldObj, "parentWorld", newObj, "boneFromAttachment");
		hkClassMemberAccessor oldBoneIndex(oldObj, "boneIndex");
		hkClassMemberAccessor newBoneIndex(newObj, "boneIndex");
		newBoneIndex.asInt16() = (hkInt16)oldBoneIndex.asInt32();
	}

	static void HK_CALL DefaultExtractedMotion( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& )
	{
		hkVersionUtil::renameMember( oldObj, "period", newObj, "duration");
		hkVersionUtil::renameMember( oldObj, "motionTrack", newObj, "referenceFrameSamples");
	}

	static void HK_CALL DeltaCompressedAnimation(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& params)
	{
		hkClassMemberAccessor oldNumberOfPoses(oldObj, "numberOfPoses");
		hkClassMemberAccessor newNumberOfPoses(newObj, "numberOfPoses");
		if( oldNumberOfPoses.isOk() && newNumberOfPoses.isOk() )
		{
			newNumberOfPoses.asInt32() = oldNumberOfPoses.asInt16();
		}
		hkClassMemberAccessor oldBlockSize(oldObj, "blockSize");
		hkClassMemberAccessor newBlockSize(newObj, "blockSize");
		if( oldBlockSize.isOk() && newBlockSize.isOk() )
		{
			newBlockSize.asInt32() = oldBlockSize.asInt16();
		}
		SkeletalAnimation(oldObj, newObj, params);
	}

	static void HK_CALL InterleavedAnimation( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& params)
	{
		hkVersionUtil::renameMember( oldObj, "poses", newObj, "transforms");
		SkeletalAnimation(oldObj, newObj, params);
	}

	static void HK_CALL MeshBinding( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor oldMapping(oldObj, "mapping");
		hkClassMemberAccessor newMappings(newObj, "mappings");
		if( oldMapping.isOk() && newMappings.isOk() )
		{
			hkClassMemberAccessor::SimpleArray* mapping = hkAllocate<hkClassMemberAccessor::SimpleArray>(1, HK_MEMORY_CLASS_EXPORT);
			tracker.addAllocation(mapping);
			// careful - old and new arrays share same memory
			hkClassMemberAccessor::SimpleArray& oldArray = oldMapping.asSimpleArray();
			mapping->data = oldArray.data;
			mapping->size = oldArray.size;

			hkClassMemberAccessor::SimpleArray& newArray = newMappings.asSimpleArray();
			newArray.data = mapping;
			newArray.size = 1;
		}
	}

	static void HK_CALL Node( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& )
	{
		hkVersionUtil::renameMember( oldObj, "keys", newObj, "keyFrames");
	}

	static void HK_CALL Skeleton(hkVariant& oldObj,hkVariant& newObj, hkObjectUpdateTracker& tracker)
	{
		hkVersionUtil::renameMember( oldObj, "hierarchy", newObj, "parentIndices");

		// Set up the array to hold the transforms
		{
			hkClassMemberAccessor bones(oldObj, "bones");
			hkClassMemberAccessor referencePose( newObj, "referencePose");
			if( bones.isOk() && referencePose.isOk() )
			{
				SimpleArray<void*>& bonesArray = *(SimpleArray<void*>*)bones.asRaw();
				SimpleArray<hkQsTransform>& referencePoseArray = *(SimpleArray<hkQsTransform>*)referencePose.asRaw();

				referencePoseArray.size = bonesArray.size;
				referencePoseArray.data = hkAllocate<hkQsTransform>( referencePoseArray.size, HK_MEMORY_CLASS_ANIM_RIG );
				tracker.addAllocation( referencePoseArray.data );

				int transformOffset = bones.getClassMember().getStructClass().getMemberByName("initialTransform")->getOffset();
				for (int i=0; i < bonesArray.size; i++)
				{
					void* original = bonesArray.data[i];
					referencePoseArray.data[i] = *(hkQsTransform*)(static_cast<char*>(original) + transformOffset);
				}
			}
		}
	}

	static void HK_CALL WaveletCompressedAnimation(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& params)
	{
		// not inherited, but has same changes
		DeltaCompressedAnimation(oldObj, newObj, params);
	}

	#define REMOVED(TYPE) { 0,0, hkVersionRegistry::VERSION_REMOVED, TYPE, HK_NULL }
	#define BINARY_IDENTICAL(OLDSIG,NEWSIG,TYPE) { OLDSIG, NEWSIG, hkVersionRegistry::VERSION_MANUAL, TYPE, HK_NULL }

	static const hkVersionRegistry::ClassAction s_updateActions[] =
	{
		// Physics

		{ 0x12e2e514, 0xda2ae69f, hkVersionRegistry::VERSION_INPLACE,	"hkConstraintInstance", HK_NULL },  // enum value added, entities type changed
		{ 0xc1b1d058, 0xa7ff561d, hkVersionRegistry::VERSION_COPY,		"hkConstrainedSystemFilter", HK_NULL}, // otherFilter member added
		{ 0x3aeda872, 0x8e3d544b, hkVersionRegistry::VERSION_COPY,		"hkCharacterProxyCinfo", HK_NULL  }, // many added members
		{ 0x4f95ebbf, 0x0bdb4a79, hkVersionRegistry::VERSION_INPLACE,	"hkWorldObject", HK_NULL },
		{ 0x88cc1e3c, 0x02a706ea, hkVersionRegistry::VERSION_COPY,		"hkPoweredHingeConstraintData", HK_NULL },
		{ 0xb3030cd9, 0x9e3807df, hkVersionRegistry::VERSION_COPY,		"hkPrismaticConstraintData", HK_NULL },
		{ 0x8f01e377, 0x09905d3a, hkVersionRegistry::VERSION_INPLACE,	"hkRagdollConstraintData", RagdollConstraintData },
		{ 0x5549062d, 0x66a50433, hkVersionRegistry::VERSION_COPY,		"hkStorageMeshShapeSubpartStorage", HK_NULL },
		{ 0xf2740354, 0xef940b7c, hkVersionRegistry::VERSION_INPLACE,	"hkWorldCinfo", WorldCinfo },
		{ 0x393ebf0f, 0x51a73ef8, hkVersionRegistry::VERSION_COPY,		"hkMeshShapeSubpart", HK_NULL },
		{ 0x5042881b, 0xa5bd1d6e, hkVersionRegistry::VERSION_COPY,		"hkConvexPieceStreamData", HK_NULL }, // now inherits from ref object

		BINARY_IDENTICAL(0x87840b54, 0x6c787842, "hkCylinderShape"),
		BINARY_IDENTICAL(0xee66859f, 0xd9784f0e, "hkKeyframedRigidMotion"),
		{ 0x2b10429e, 0x9eeaa6f5, hkVersionRegistry::VERSION_COPY,		"hkMeshShape", HK_NULL }, //TODO: m_subparts member copies (hkMeshShapeSubpart)*/
		//BINARY_IDENTICAL(0x2b10429e, 0x9eeaa6f5, "hkMeshShape"),
		BINARY_IDENTICAL(0x2eaeb17f, 0x1ff447b3, "hkClass"),
		BINARY_IDENTICAL(0xa05db6ee, 0x9ce308e9, "hkProperty"),

		REMOVED("hkBlendingMotor"),
		REMOVED("hkSpringDamperMotor"),
		REMOVED("hkStrongestMotor"),
		REMOVED("hkVelocityMotor"),
		REMOVED("hkMotorController"),
		REMOVED("hkSampledHeightFieldBase"),

		// Animation

		{ 0xe59b224f, 0xef964a74, hkVersionRegistry::VERSION_INPLACE, "hkAnimationBinding", AnimationBinding },
		{ 0xbbbca03d, 0xa74011f0, hkVersionRegistry::VERSION_COPY,    "hkBone", Bone },
		{ 0x64e34460, 0x9dd3289c, hkVersionRegistry::VERSION_COPY | hkVersionRegistry::VERSION_VARIANT,    "hkBoneAttachment", BoneAttachment },
		{ 0x0c33bac4, 0x122f506b, hkVersionRegistry::VERSION_COPY,    "hkDefaultExtractedMotion", DefaultExtractedMotion },
		{ 0x9e581f7b, 0x2c3bd732, hkVersionRegistry::VERSION_COPY,    "hkDeltaCompressedAnimation", DeltaCompressedAnimation },
		{ 0x45aeafd9, 0xc21c54ff, hkVersionRegistry::VERSION_COPY,    "hkInterleavedAnimation", SkeletalAnimation },
		{ 0x75c267ef, 0x88f9319c, hkVersionRegistry::VERSION_INPLACE, "hkMeshBinding", MeshBinding },
		{ 0x4f3dea6e, 0x4af7c559, hkVersionRegistry::VERSION_COPY,    "hkSkeleton", Skeleton },	
		{ 0x0ddead27, 0xce906bbf, hkVersionRegistry::VERSION_COPY,    "hkWaveletCompressedAnimation", SkeletalAnimation },
		{ 0x11e05c14, 0xb1aac849, hkVersionRegistry::VERSION_INPLACE, "hkSkeletalAnimation", SkeletalAnimation },

		REMOVED("hkRigidAccumulator"),
		REMOVED("hkSolverResults"),
		REMOVED("hkVelocityAccumulator"),

		// Complete

		{ 0x17c284fb, 0x2c8afd32, hkVersionRegistry::VERSION_COPY, "hkRagdollInstance", RagdollInstance }, // offsets, members added

		// Scenedata
		{ 0x35e1060e, 0x35e1060e, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainerNamedVariant", HK_NULL},
		{ 0x12a4e063, 0x12a4e063, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainer", HK_NULL},
		{ 0xe085ba9f, 0xe085ba9f, hkVersionRegistry::VERSION_VARIANT, "hkxMaterialTextureStage", HK_NULL},
		{ 0xd54f2307, 0xba07287d, hkVersionRegistry::VERSION_VARIANT | hkVersionRegistry::VERSION_MANUAL, "hkxNode", Node }, // hkxNode::keys renamed to hkxNode::keyFrames
		{ 0x2d5d1d38, 0x9b712385, hkVersionRegistry::VERSION_VARIANT | hkVersionRegistry::VERSION_COPY, "hkxMaterial", HK_NULL }, // name added

		{ 0x57061454, 0x57061454, hkVersionRegistry::VERSION_HOMOGENEOUSARRAY, "hkxVertexBuffer", HK_NULL },

		{ 0,0, 0, HK_NULL, HK_NULL }
	};

	static const hkVersionRegistry::ClassRename s_renames[] =
	{
		{ "hkDefaultExtractedMotion", "hkDefaultAnimatedReferenceFrame" },
		{ "hkDeltaCompressedAnimation", "hkDeltaCompressedSkeletalAnimation" },
		{ "hkDeltaCompressedAnimationQuantizationFormat", "hkDeltaCompressedSkeletalAnimationQuantizationFormat" },
		{ "hkInterleavedAnimation", "hkInterleavedSkeletalAnimation" },
		{ "hkWaveletCompressedAnimation", "hkWaveletSkeletalAnimation" },		
		{ "hkWaveletCompressedAnimationQuantizationFormat", "hkWaveletSkeletalAnimationQuantizationFormat" },
		{ "hkAnimationTrack", "hkAnnotationTrack" },
		{ "hkAnimationTrackAnnotation", "hkAnnotationTrackAnnotation" },
		{ "hkExtractedMotion", "hkAnimatedReferenceFrame" },
		{ HK_NULL, HK_NULL }
	};

#define HK_COMPAT_VERSION_FROM hkHavok300Classes
#define HK_COMPAT_VERSION_TO hkHavok310Classes
#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO

} // namespace hkCompat_hk300_hk310

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
