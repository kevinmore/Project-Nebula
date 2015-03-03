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
#include <Common/Compat/Deprecated/Version/hkVersionUtilOld.h>
#include <Common/Serialize/Version/hkVersionUtil.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>
#include <Common/Base/hkBase.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>

#define HK_COMPAT_VERSION_FROM hkHavok700b1Classes
#define HK_COMPAT_VERSION_TO hkHavok700r1Classes

namespace HK_COMPAT_VERSION_FROM
{
	extern hkClass hkaSkeletonClass;
	extern hkClass hkaAnimationClass;
	extern hkClass hkReferencedObjectClass;
	extern hkClass hkRootLevelContainerNamedVariantClass;
}

namespace hkCompat_hk700b1_hk700r1
{

#define REMOVED(TYPE) { 0,0, hkVersionRegistry::VERSION_REMOVED, TYPE, HK_NULL }
#define BINARY_IDENTICAL(OLDSIG,NEWSIG,TYPE) { OLDSIG, NEWSIG, hkVersionRegistry::VERSION_MANUAL, TYPE, HK_NULL }

	struct DummyArray
	{
		void* data;
		int size;
		int capacity;
	};

	struct Context
	{
		Context(hkArray<hkVariant>& objects, hkObjectUpdateTracker& tracker, const hkVersionRegistry::UpdateDescription& versionUpdateDescription)
			: m_objects(objects), m_tracker(tracker), m_versionUpdateDescription(versionUpdateDescription)
		{
			HK_ASSERT(0x771f058c, s_context == HK_NULL);
			s_context = this;
		}

		~Context()
		{
			removeAllOldSkeletonBones();
			cleanupAnnotationTracks();
			s_context = HK_NULL;
		}

		void collectOldSkeletonBone(void* bonePointer)
		{
			HK_ASSERT(0x574ead51, bonePointer);
			m_oldSkeletonBones.pushBack(bonePointer);
		}

		const hkClass* findNewClassFromOld(const hkClass& oldClass)
		{
			const hkClassNameRegistry* classReg = hkVersionRegistry::getInstance().getClassNameRegistry(m_versionUpdateDescription.m_newClassRegistry->getName());
			HK_ASSERT(0x5bd053bf, classReg);
			const char* name = oldClass.getName();
			const hkVersionRegistry::UpdateDescription* desc = &m_versionUpdateDescription;
			while( desc )
			{
				// check if class is removed -> return HK_NULL as we may find new unrelated class with the same name
				const hkVersionRegistry::ClassAction* actions = desc->m_actions;
				for( int i = 0; actions[i].oldClassName != HK_NULL; ++i )
				{
					if( hkString::strCmp(actions[i].oldClassName, oldClass.getName()) == 0 )
					{
						if( actions[i].versionFlags & hkVersionRegistry::VERSION_REMOVED )
						{
							return HK_NULL;
						}
						break;
					}
				}
				// class is not removed, check if it was renamed
				const hkVersionRegistry::ClassRename* renames = desc->m_renames;
				for( int i = 0; renames[i].oldName != HK_NULL; ++i )
				{
					if( hkString::strCmp(renames[i].oldName, name) == 0 )
					{
						return classReg->getClassByName(renames[i].newName);
					}
				}
				desc = desc->m_next;
			}
			return classReg->getClassByName(name);
		}

		void removeAllOldSkeletonBones()
		{
			for( int i = 0; i < m_oldSkeletonBones.getSize(); ++i )
			{
				void* oldBone = m_oldSkeletonBones[i];
				HK_ON_DEBUG(bool foundBone = false);
				for( int j = 0; j < m_objects.getSize(); ++j )
				{
					hkVariant& v = m_objects[j];
					if( v.m_object == oldBone )
					{
						HK_ON_DEBUG(foundBone = true);
						HK_ASSERT(0x2659d96e, hkString::strCmp(v.m_class->getName(), "hkaBone") == 0);
						// The bones in skeleton must not be shared, remove it
						m_tracker.replaceObject(v.m_object, HK_NULL, HK_NULL);
						m_objects.removeAt(j);
						break;
					}
				}
				HK_ASSERT(0x2cf6b600, foundBone ); // the bones are not binary identical?
			}
		}

		void putSkeletonObjectsLast()
		{
			hkArray<hkVariant> skeletonObjects;
			for( int i = m_objects.getSize() - 1; i >=0; --i )
			{
				hkVariant& v = m_objects[i];
				if( hkString::strCmp(v.m_class->getName(), HK_COMPAT_VERSION_FROM::hkaSkeletonClass.getName()) == 0 )
				{
					skeletonObjects.pushBack(v);
					m_objects.removeAt(i);
				}
			}
			m_objects.spliceInto(m_objects.getSize(), 0, skeletonObjects.begin(), skeletonObjects.getSize());
		}

		void collectOldAnnotationTracks(void* old)
		{
			m_oldAnnotationTracks.pushBack(old);
		}

		void cleanupAnnotationTracks()
		{
			for( int i = 0; i < m_oldAnnotationTracks.getSize(); ++i )
			{
				void* newTrack = getNewObjectFromOld( m_oldAnnotationTracks[i] );
				HK_ASSERT(0x2659d96d, newTrack);
				//HK_ON_DEBUG(bool foundTrack = false);
				for( int j = 0; j < m_objects.getSize(); ++j )
				{
					hkVariant& v = m_objects[j];
					if( v.m_object == newTrack )
					{
						//HK_ON_DEBUG(foundTrack = true);

						HK_ASSERT(0x2659d96f, hkString::strCmp(v.m_class->getName(), "hkaAnnotationTrack") == 0);
						// The new annotation tracks in animation must not be shared, remove it
						m_tracker.replaceObject(v.m_object, HK_NULL, HK_NULL);
						m_objects.removeAt(j);
						break;
					}
				}
			}
		}

		void collectOldNewObjects(void* oldObj, void* newObj)
		{
            m_newObjectFromOld.insert(oldObj, newObj);
		}

		void* getNewObjectFromOld(void* oldObj)
		{
            return m_newObjectFromOld.getWithDefault(oldObj, HK_NULL);
		} 

		void putAnimationObjectsLast()
		{
            hkArray<hkVariant> animationObjects;
            for( int i = m_objects.getSize() - 1; i >=0; --i )
            {
				hkVariant& v = m_objects[i];
				if( hkHavok700b1Classes::hkaAnimationClass.isSuperClass( *v.m_class ) )
				{
					animationObjects.pushBack(v);
					m_objects.removeAt(i);
				}
            }
            m_objects.spliceInto(m_objects.getSize(), 0, animationObjects.begin(),animationObjects.getSize());
		}

		void registerNewObjectForVersioning(const hkVariant& v)
		{
			m_tracker.addFinish(v.m_object, v.m_class->getName());
			m_objects.pushBack(v);
		}

		static Context* s_context;

	private:
		hkArray<hkVariant>& m_objects;
		hkObjectUpdateTracker& m_tracker;
		hkArray<void*> m_oldSkeletonBones;

		hkPointerMap<void*, void*> m_newObjectFromOld; 
		hkArray<void*> m_oldAnnotationTracks;
		const hkVersionRegistry::UpdateDescription& m_versionUpdateDescription;

	};

	Context* Context::s_context = HK_NULL;

	static void Update_ignore( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
	}

	static void Update_assert( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		HK_ASSERT2(0xad904271, false, "This object is not expected to be updated directly.");
	}

	static void Update_NotImplemented( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		HK_ASSERT2(0xad901971, false, "Not implemented.");
	}

	static void Update_Ai_Ignore( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		HK_ASSERT2(0xf237846a, false, "Ai class versioning for packfiles is not supported. Use tagfiles instead");
	}

	// CLOTH

	// Old enum
	enum
	{
		HCL_NO_ACCESS = 0,
		HCL_POSITION_ACCESS_READ_ONLY = 1,
		HCL_POSITION_ACCESS_WRITE_ONLY = 2,
		HCL_POSITION_ACCESS_WRITE_ALL = 4,
		HCL_POSITION_ACCESS_READ_WRITE = (HCL_POSITION_ACCESS_READ_ONLY | HCL_POSITION_ACCESS_WRITE_ONLY), // 3, Slowest
		HCL_POSITION_ACCESS_MASK = (HCL_POSITION_ACCESS_READ_WRITE | HCL_POSITION_ACCESS_WRITE_ALL),
	};

	// New per-component enum
	enum
	{
		USAGE_NONE = 0,
		USAGE_READ = 1,
		USAGE_WRITE = 2,
		USAGE_FULL_WRITE = 4,
		USAGE_READ_BEFORE_WRITE = 8,
	};


	// Cloth State / Cloth State Buffer Access
	static void Update_hclClothStateBufferAccess (hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker)
	{
		{
			hkClassMemberAccessor accessFlagsIn(oldVar, "accessFlags");
			hkClassMemberAccessor bufferUsageOut(newVar, "bufferUsage");

			const hkUint32 accessFlags = accessFlagsIn.asUint32();

			// 4 components
			for (int i=0; i<4; i++)
			{
				// Unfortunately, bitangents skipped bit 9!
				const int shift = (i==3) ? 10 : (i*3);
				int oldComponentFlags = (accessFlags >> shift) & HCL_POSITION_ACCESS_MASK;

				hkUint8 newComponentFlags = 0;
				if (oldComponentFlags & HCL_POSITION_ACCESS_READ_ONLY)
				{
					newComponentFlags |= USAGE_READ;
				}
				if (oldComponentFlags & HCL_POSITION_ACCESS_WRITE_ONLY)
				{
					newComponentFlags |= USAGE_WRITE;
				}
				if (oldComponentFlags & HCL_POSITION_ACCESS_READ_WRITE)
				{
					newComponentFlags |= USAGE_READ_BEFORE_WRITE; // Assume the worst
				}
				if (oldComponentFlags & HCL_POSITION_ACCESS_WRITE_ALL)
				{
					newComponentFlags |= USAGE_FULL_WRITE;
				}

				bufferUsageOut.object().member("perComponentFlags").asUint8(i) = newComponentFlags;

			}

			// Assume no need for triangles
			bufferUsageOut.object().member("trianglesRead").asBool()= false;
			
		}
	
	}

	// We didn't really change ClothState but we changed ClothState::BufferAccess
	static void Update_hclClothState (hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor oldArray(oldVar, "usedBuffers");
		const hkClassMemberAccessor::SimpleArray& simpleArrayOld = oldArray.asSimpleArray();


		hkClassMemberAccessor newArray(newVar, "usedBuffers");
		const hkClassMemberAccessor::SimpleArray& simpleArrayNew = newArray.asSimpleArray();

		HK_ASSERT(0x224e444f, simpleArrayNew.size == simpleArrayOld.size);

		for (int i=0; i<simpleArrayNew.size; ++i)
		{
			const hkClass* oldBufferAccessClass = oldArray.getClassMember().getClass();
			HK_ASSERT(0x4858aabb, oldBufferAccessClass);
			hkVariant oldUsedBuffItem = {hkAddByteOffset(simpleArrayOld.data, i*oldBufferAccessClass->getObjectSize()), oldBufferAccessClass};

			const hkClass* newBufferAccessClass = newArray.getClassMember().getClass();
			HK_ASSERT(0x4858aabb, newBufferAccessClass);
			hkVariant newUsedBuffItem = {hkAddByteOffset(simpleArrayNew.data, i*newBufferAccessClass->getObjectSize()), newBufferAccessClass};

			Update_hclClothStateBufferAccess (oldUsedBuffItem, newUsedBuffItem, tracker);
		}

	}

	static void Update_hclSkinOperator (hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker)
	{
		const hkUint32 startVertex = hkClassMemberAccessor(oldVar, "startVertex").asInt32();
		const hkUint32 endVertex = hkClassMemberAccessor(oldVar, "endVertex").asInt32();

		const hkUint32 numVerts = endVertex - startVertex;

		hkClassMemberAccessor influencesAcc (oldVar, "boneInfluenceStartPerVertex");
		const hkUint16* influences = reinterpret_cast<hkUint16*> (influencesAcc.asSimpleArray().data);
		bool partial = false;
		for (hkUint32 i=0; i<numVerts; ++i)
		{
			const int startEntry = influences[i];
			const int endEntry = influences[i+1];
			const int numEntries = endEntry-startEntry;

			if (numEntries==0)
			{
				partial =true;
				break;
			}
		}

		hkClassMemberAccessor(newVar, "partialSkinning").asBool() = partial;
	}

	static void Update_hclGatherAllVerticesOperator (hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor viFromVoAcc (oldVar, "vertexInputFromVertexOutput");
		const int numEntries = viFromVoAcc.asSimpleArray().size;
		const hkInt32* viFromVo = reinterpret_cast<hkInt32*> (viFromVoAcc.asSimpleArray().data);

		bool partial = false;
		for (int i=0; i<numEntries; ++i)
		{
			if (viFromVo[i]<0)
			{
				partial =true;
				break;
			}
		}

		hkClassMemberAccessor(newVar, "partialGather").asBool() = partial;
	}

	static void Update_hclMeshMeshDeformOperator (hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker)
	{
		const hkUint32 startVertex = hkClassMemberAccessor(oldVar, "startVertex").asInt32();
		const hkUint32 endVertex = hkClassMemberAccessor(oldVar, "endVertex").asInt32();

		const hkUint32 numVerts = endVertex - startVertex;

		hkClassMemberAccessor influencesAcc (oldVar, "triangleVertexStartForVertex");
		const hkUint16* influences = reinterpret_cast<hkUint16*> (influencesAcc.asSimpleArray().data);
		bool partial = false;
		for (hkUint32 i=0; i<numVerts; ++i)
		{
			const int startEntry = influences[i];
			const int endEntry = influences[i+1];
			const int numEntries = endEntry-startEntry;

			if (numEntries==0)
			{
				partial =true;
				break;
			}
		}

		hkClassMemberAccessor(newVar, "partialDeform").asBool() = partial;
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
			hkString::memSet( dummyArray.data, 0, numBytes );
		}
	}


	static void Update_hclStorageSetupMeshSection(hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor oldVertexArray ( oldVar, "vertices");
		const int numSectionVertices = oldVertexArray.asSimpleArray().size;

		// Adding a new array to store normal IDs, initialized to the default of a unique ID per vertex in the section.
		hkClassMemberAccessor newNormalIDsArray ( newVar, "normalIDs");

		// allocate memory for the array
		initArray( newNormalIDsArray, numSectionVertices, tracker);

		// Set the normal IDs
		for (int i=0; i<numSectionVertices; ++i)
		{
			reinterpret_cast<hkUint16*> (newNormalIDsArray.asSimpleArray().data)[i] = static_cast<hkUint16>(i);
		}
	}

	static void Update_hkbDemoConfig(hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor oldStickVariables( oldVar, "stickVariables" );
		hkClassMemberAccessor newStickVariables( newVar, "stickVariables" );

		// We'll re-use the memory from the old object since it was a fixed c-style array
		DummyArray* newArray = (DummyArray*)(newStickVariables.getAddress());
		newArray->data = oldStickVariables.getAddress();
		newArray->capacity = 12 | int(0x80000000); // DONT_DEALLOCATE_FLAG
		newArray->size = 12;
	}

static const hkVersionRegistry::ClassAction s_updateActions[] =
{
	// variants
	{ 0x1388d601, 0x1388d601, hkVersionRegistry::VERSION_VARIANT, "hkCustomAttributesAttribute", HK_NULL },
	{ 0xbff19005, 0xbff19005, hkVersionRegistry::VERSION_VARIANT, "hkCustomAttributes", HK_NULL },
	{ 0x15d99dc6, 0x27812d8d, hkVersionRegistry::VERSION_VARIANT | hkVersionRegistry::VERSION_COPY, "hkbVariableValueSet", HK_NULL },

	// Common
	BINARY_IDENTICAL(0x11545121, 0xe758f63c, "hkSimpleLocalFrame"),

	// Destruction
	{ 0x7325d9bc, 0x5287e598, hkVersionRegistry::VERSION_COPY, "hkdCompoundBreakableShape", HK_NULL },
	{ 0x76658b24, 0x4c96dc25, hkVersionRegistry::VERSION_COPY, "hkdBreakableShape", HK_NULL },
	{ 0x335c4983, 0x6a5c181b, hkVersionRegistry::VERSION_COPY, "hkdGeometry", HK_NULL },
	{ 0x62680e5,  0xdd6de80b, hkVersionRegistry::VERSION_COPY, "hkdMeshSimplifierAction", HK_NULL },
	{ 0x45da6b5d, 0x7fa1be83, hkVersionRegistry::VERSION_COPY, "hkdSetRigidBodyPropertiesAction", HK_NULL },

	// Physics
	BINARY_IDENTICAL(0x6de8f8e, 0x4fb4e6bb, "hkpConstraintInstance"),
	{ 0x8944dc00, 0xc7bbc095, hkVersionRegistry::VERSION_COPY, "hkpWorldCinfo", HK_NULL }, // HVK-5170 Added new member.
	{ 0x80d3a7da, 0x1ce00a45, hkVersionRegistry::VERSION_COPY, "hkpCompressedMeshShape", HK_NULL },
	BINARY_IDENTICAL(0xc1afa146, 0xe92b83be, "hkpBridgeConstraintAtom"),
	BINARY_IDENTICAL(0xab70d590, 0x7f52f378, "hkpBallSocketChainData"),
	BINARY_IDENTICAL(0xdf812dd4, 0x7034bdb6, "hkpBreakableConstraintData"),
	BINARY_IDENTICAL(0x65cd952a, 0x0170ede1, "hkpBridgeAtoms"),
	BINARY_IDENTICAL(0x9f23b5de, 0x64ebd59e, "hkpGenericConstraintData"),
	BINARY_IDENTICAL(0x5ac85b1a, 0xb9e22b1b, "hkpMalleableConstraintData"),
	BINARY_IDENTICAL(0x06db0556, 0xa8ab80bf, "hkpPointToPathConstraintData"),
	BINARY_IDENTICAL(0xa8ad141f, 0xc7c44da8, "hkpPoweredChainData"),
	BINARY_IDENTICAL(0x3066daaa, 0xd41dc47e, "hkpStiffSpringChainData"),
	{ 0xb8168996, 0x7aa88333, hkVersionRegistry::VERSION_COPY, "hkpExtendedMeshShape", HK_NULL },
	{ 0xad025546, 0x9c94f8b3, hkVersionRegistry::VERSION_COPY, "hkpExtendedMeshShapeShapesSubpart", HK_NULL },
	{ 0xd7628aa1, 0x3f7d804c, hkVersionRegistry::VERSION_COPY, "hkpStorageExtendedMeshShapeShapeSubpartStorage", HK_NULL },
	{ 0x00000000, 0x00000000, hkVersionRegistry::VERSION_COPY, "hkpSerializedAgentNnEntry", HK_NULL }, // bump internal buffer up 128->160(for 64 bit)

	// Animation

	// Behavior
	{ 0x2c76ae79, 0x4479ab06, hkVersionRegistry::VERSION_COPY, "hkbBehaviorGraph", HK_NULL },
	{ 0x6ebb687b, 0xcd83778f, hkVersionRegistry::VERSION_COPY, "hkbCharacterData", HK_NULL },
	{ 0x25e7e488, 0x79235713, hkVersionRegistry::VERSION_COPY, "hkbCharacterSetup", HK_NULL },
	{ 0x5d2639e5, 0x1433e0a , hkVersionRegistry::VERSION_COPY, "hkbCharacterStringData", HK_NULL },
	{ 0xb86510d6, 0x5240cd73, hkVersionRegistry::VERSION_COPY, "hkbClipGenerator", HK_NULL },
	{ 0x66ff988e, 0x241cee4f, hkVersionRegistry::VERSION_COPY, "hkbContext", HK_NULL },
	BINARY_IDENTICAL( 0x3a2f4672, 0xe89ce125, "hkbCompiledExpressionSet" ),
	BINARY_IDENTICAL( 0x5f860541, 0x539aa3ab, "hkbCompiledExpressionSetToken" ),
	{ 0x2bbef407, 0xdf8b686c, hkVersionRegistry::VERSION_COPY, "hkbDemoConfig", Update_hkbDemoConfig },
	BINARY_IDENTICAL( 0xfc825fae, 0xcebe6db0, "hkbDemoConfigStickVariableInfo" ),
	{ 0xef747f76, 0x4200a9c4, hkVersionRegistry::VERSION_COPY, "hkbFootIkModifier", HK_NULL },
	{ 0xc7b16861, 0xe53cf475, hkVersionRegistry::VERSION_COPY, "hkbPoseMatchingGenerator", HK_NULL },
	{ 0x8b625cc0, 0x5089a395, hkVersionRegistry::VERSION_COPY, "hkbPoweredRagdollControlsModifier", HK_NULL },
	{ 0xe6f25897, 0x66440c5b, hkVersionRegistry::VERSION_COPY, "hkbRagdollController", HK_NULL },
	{ 0x750a2d67, 0xd067a419, hkVersionRegistry::VERSION_COPY, "hkbRigidBodyRagdollControlsModifier", HK_NULL },
	{ 0xa96e7a21, 0x7de5ca97, hkVersionRegistry::VERSION_COPY, "hkbSimpleCharacter", HK_NULL },
	{ 0x60166a1f, 0x77ebb964, hkVersionRegistry::VERSION_COPY, "hkbStateMachine", HK_NULL },
	{ 0x24bb58af, 0x204be7e3, hkVersionRegistry::VERSION_COPY, "hkbGetUpModifier", HK_NULL },
	REMOVED("hkbBalanceControllerModifier"),
	REMOVED("hkbRagdollDriverModifier"),
	REMOVED("hkbRagdollModifier"),

	// Cloth
	{ 0xc8919997, 0xb620947b, hkVersionRegistry::VERSION_COPY, "hclClothState", Update_hclClothState },
	// This compat function is not called directly - it is called by Update_hclClothState
	{ 0x4d38a6aa, 0x6a9c903, hkVersionRegistry::VERSION_COPY, "hclClothStateBufferAccess", Update_hclClothStateBufferAccess},
	BINARY_IDENTICAL( 0x587b31bc, 0xe36985b8, "hclRuntimeConversionInfo"),
	BINARY_IDENTICAL( 0xb4a45687, 0xa19a443a, "hclRuntimeConversionInfoElementConversion"),
	BINARY_IDENTICAL( 0xa715d259, 0x904c0e80, "hclInputConvertOperator"),
	BINARY_IDENTICAL( 0xa715d259, 0x904c0e80, "hclOutputConvertOperator"),
	{ 0x415239c9, 0xbe929277, hkVersionRegistry::VERSION_COPY, "hclSkinOperator", Update_hclSkinOperator},
	{ 0xc6bf70c2, 0x57eaf696, hkVersionRegistry::VERSION_COPY, "hclGatherAllVerticesOperator", Update_hclGatherAllVerticesOperator},
	{ 0x702feaac, 0xf9ae608e, hkVersionRegistry::VERSION_COPY, "hclMeshMeshDeformOperator", Update_hclMeshMeshDeformOperator},
	{ 0xe667504e, 0x80f8175d , hkVersionRegistry::VERSION_COPY, "hclStorageSetupMeshSection", Update_hclStorageSetupMeshSection },
	{ 0x2c66aa23, 0xb62ce4fa , hkVersionRegistry::VERSION_COPY, "hclSimClothData", HK_NULL },
	{ 0xb42d024a, 0x4fab7080 , hkVersionRegistry::VERSION_COPY, "hclSimClothDataSimulationInfo", HK_NULL },
	BINARY_IDENTICAL( 0xd5b4fc3d, 0x3750d9b9, "hclNamedTransformSetSetupObject" ),

	{ 0, 0, 0, HK_NULL, HK_NULL }
};	 

static const hkVersionRegistry::ClassRename s_renames[] =
{
	{ HK_NULL, HK_NULL }
};

#define HK_COMPAT_OPTIONAL_UPDATE_FUNC update

extern hkVersionRegistry::UpdateDescription hkVersionUpdateDescription;

static hkResult HK_CALL update(
							   hkArray<hkVariant>& objectsInOut,
							   hkObjectUpdateTracker& tracker )
{
	Context context(objectsInOut, tracker, hkVersionUpdateDescription);

	context.putSkeletonObjectsLast();
	context.putAnimationObjectsLast();

	return hkVersionUtil::updateSingleVersion( objectsInOut, tracker, hkVersionUpdateDescription, hkVersionRegistry::getInstance().getClassNameRegistry(HK_COMPAT_VERSION_TO::VersionString) );
}

#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO

} // namespace hkCompat_hk700b1_hk700r1

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
