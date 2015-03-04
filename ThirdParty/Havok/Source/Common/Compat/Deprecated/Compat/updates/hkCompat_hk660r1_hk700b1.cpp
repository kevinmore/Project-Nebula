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
#include <Common/Compat/Deprecated/Version/hkVersionUtilOld.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>
#include <Common/Base/hkBase.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>

#define HK_COMPAT_VERSION_FROM hkHavok660r1Classes
#define HK_COMPAT_VERSION_TO hkHavok700b1Classes

namespace HK_COMPAT_VERSION_FROM
{
	extern hkClass hkaSkeletonClass;
	extern hkClass hkaAnimationClass;
	extern hkClass hkReferencedObjectClass;
	extern hkClass hkRootLevelContainerNamedVariantClass;
}

namespace hkCompat_hk660r1_hk700b1
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
				if( hkHavok660r1Classes::hkaAnimationClass.isSuperClass( *v.m_class ) )
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

	// allocate the memory for an array
	static void initArray( const hkClassMemberAccessor& member, int numElements, hkObjectUpdateTracker& tracker )
	{
		DummyArray& dummyArray = *static_cast<DummyArray*>(member.getAddress());
		dummyArray.size = numElements;
		dummyArray.capacity = hkArray<char>::DONT_DEALLOCATE_FLAG | numElements;

		if( numElements > 0 )
		{
			int numBytes = numElements * member.getClassMember().getArrayMemberSize();
			dummyArray.data = hkAllocateChunk<char>( numBytes, HK_MEMORY_CLASS_ARRAY );
			tracker.addChunk( dummyArray.data, numBytes, HK_MEMORY_CLASS_SERIALIZE );
		}
	}
	
	static void Update_hkaAnimationPreviewColorContainer( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// Does nothing
	}

	static void Update_hkaSkeleton( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// convert array of hkaBone pointers to array of structs
		hkClassMemberAccessor oldBones(oldObj, "bones");
		hkClassMemberAccessor newBones(newObj, "bones");

		const hkClassMemberAccessor::SimpleArray& oldBonesArray = oldBones.asSimpleArray();
		initArray(newBones, oldBonesArray.size, tracker);
		hkClassMemberAccessor::SimpleArray& newBonesArray = newBones.asSimpleArray();
		HK_ASSERT(0x4858a23b, oldBones.getClassMember().getClass() && newBones.getClassMember().getClass()
			&& oldBones.getClassMember().getStructClass().getObjectSize() == newBones.getClassMember().getStructClass().getObjectSize());
		HK_ASSERT(0x2edafef6, newBones.getClassMember().getStructClass().getSignature() == 0x35912f8a); // hkaBone is binary identical, no manual updates
		int boneSize = newBones.getClassMember().getStructClass().getObjectSize();
		for( int i = 0; i < oldBonesArray.size; ++i )
		{
			void* oldBone = static_cast<void**>(oldBonesArray.data)[i];
			Context::s_context->collectOldSkeletonBone(oldBone);
			hkMemUtil::memCpy(static_cast<char*>(newBonesArray.data)+i*boneSize, oldBone, boneSize);
		}
	}

	static void Update_hkaAnnotationTrack( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		Context::s_context->collectOldNewObjects(oldObj.m_object, newObj.m_object);
	}

	static void Update_hkaAnimation( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// convert array of hkaBone pointers to array of structs
		hkClassMemberAccessor oldAnnotationTracks(oldObj, "annotationTracks");
		hkClassMemberAccessor newAnnotationTracks(newObj, "annotationTracks");

		const hkClassMemberAccessor::SimpleArray& oldAnnotationTracksArray = oldAnnotationTracks.asSimpleArray();
		initArray(newAnnotationTracks, oldAnnotationTracksArray.size, tracker);
		hkClassMemberAccessor::SimpleArray& newAnnotationTracksArray = newAnnotationTracks.asSimpleArray();
		int AnnotationTracksize = newAnnotationTracks.getClassMember().getStructClass().getObjectSize();
		for( int i = 0; i < oldAnnotationTracksArray.size; ++i )
		{
			void* oldAnnotationTrack = static_cast<void**>(oldAnnotationTracksArray.data)[i];
			Context::s_context->collectOldAnnotationTracks(oldAnnotationTrack);
			void* newAnnotationTrack = Context::s_context->getNewObjectFromOld( oldAnnotationTrack );
			HK_ASSERT(0x2edafef6, newAnnotationTrack != HK_NULL );
			hkMemUtil::memCpy(static_cast<char*>(newAnnotationTracksArray.data)+i*AnnotationTracksize, newAnnotationTrack, AnnotationTracksize);
		}
	}

	static void Update_hkaInterleavedUncompressedAnimation( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// Ensure the base class is updated
		Update_hkaAnimation( oldObj, newObj, tracker );
	}

	static void Update_hkaSplineCompressedAnimation( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// Ensure the base class is updated
		Update_hkaAnimation( oldObj, newObj, tracker );
	}

	static void Update_hkaDeltaCompressedAnimation( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// Ensure the base class is updated
		Update_hkaAnimation( oldObj, newObj, tracker );
	}

	static void Update_hkaWaveletCompressedAnimation( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// Ensure the base class is updated
		Update_hkaAnimation( oldObj, newObj, tracker );
	}

	static void Update_hkpEntity( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkVersionUtil::renameMember( oldObj,"processContactCallbackDelay", newObj, "contactPointCallbackDelay" );
		hkVersionUtil::renameMember( oldObj,"numUserDatasInContactPointProperties", newObj, "numShapeKeysInContactPointProperties" );
	}

	static void Update_hkbClipGenerator( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// We removed MODE_USER_CONTROLLED_LOOPING(3).
		// We need to decrement all modes higher than 3.
		// we also want to replace MODE_USER_CONTROLLED_LOOPING(3)
		// with MODE_USER_CONTROLLED(2).  This code does both.

		hkClassMemberAccessor oldMode(oldObj, "mode");
		hkClassMemberAccessor newMode(newObj, "mode");

		if ( oldMode.asInt8() < 3 )
		{
			newMode.asInt8() = oldMode.asInt8();
		}
		else
		{
			newMode.asInt8() = oldMode.asInt8() - 1;
		}
	}

	// CLOTH

	static void Update_hclClothStateBufferAccess (hkVariant& , hkVariant& newVar, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor bufferIndex(newVar, "bufferIndex");
		hkClassMemberAccessor shadowIndex(newVar, "shadowBufferIndex");

		shadowIndex.asUint32() = bufferIndex.asUint32();
	}

	static void Update_hclClothState (hkVariant& oldVar, hkVariant& newVar, hkObjectUpdateTracker& tracker)
	{
		hkClassMemberAccessor newArray(newVar, "usedBuffers");
		const hkClassMemberAccessor::SimpleArray& simpleArray = newArray.asSimpleArray();

		for (int i=0; i<simpleArray.size; ++i)
		{
			hkVariant oldUsedBuffItem = {HK_NULL, HK_NULL};
			const hkClass* newBufferAccessClass = newArray.getClassMember().getClass();
			HK_ASSERT(0x4858aabb, newBufferAccessClass);
			hkVariant newUsedBuffItem = {hkAddByteOffset(simpleArray.data, i*newBufferAccessClass->getObjectSize()), newBufferAccessClass};
			Update_hclClothStateBufferAccess (oldUsedBuffItem, newUsedBuffItem, tracker);
		}

	}

	static void Update_hkAlignSceneToNodeOptions( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldNodeName(oldObj, "nodeName");
		hkClassMemberAccessor newNodeName(newObj, "nodeName");
		hkClassMemberAccessor::SimpleArray& nameArray = oldNodeName.asSimpleArray();
		HK_ASSERT(0x3fbc22f1, sizeof(hkStringPtr) == sizeof(char*));
		newNodeName.asCstring() = static_cast<char*>(nameArray.data); // just assign the pointer, owned by packfile data
	}

	static void Update_hkxSparselyAnimatedString( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldStrings(oldObj, "strings");
		hkClassMemberAccessor newStrings(newObj, "strings");
		hkClassMemberAccessor::SimpleArray& simpleArray = oldStrings.asSimpleArray(); // simple array of char*
		hkClassMemberAccessor::SimpleArray& stringsArray = newStrings.asSimpleArray(); // array of hkStringPtr
		HK_ASSERT(0x4b2198f8, stringsArray.data == HK_NULL && stringsArray.size == 0);
		stringsArray = simpleArray; // just assign the contents, owned by packfile data
		reinterpret_cast<DummyArray&>(stringsArray).capacity = hkArray<char>::DONT_DEALLOCATE_FLAG | stringsArray.size; // set the capacity and flag
	}

	static hkVariant newObject( const hkClass& klass, hkObjectUpdateTracker& tracker )
	{
		hkVariant v;
		v.m_object = hkMemHeapBlockAlloc<void>(klass.getObjectSize());
		v.m_class = &klass;
		tracker.addChunk( v.m_object, klass.getObjectSize(), HK_MEMORY_CLASS_BASE);
		Context::s_context->registerNewObjectForVersioning(v);
		return v;
	}

	static inline void registerNewArray(DummyArray& array, int itemSize, hkObjectUpdateTracker& tracker)
	{
		if( array.data )
		{
			tracker.addChunk(array.data, array.capacity*itemSize, HK_MEMORY_CLASS_ARRAY );
			reinterpret_cast<DummyArray&>(array).capacity |= hkArray<char>::DONT_DEALLOCATE_FLAG;
		}
	}

	static void Update_hkxVertexBuffer( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldDesc( oldObj, "vertexDesc");
		hkClassMemberAccessor newDesc( newObj, "desc");
		int oldVertexStride = hkClassMemberAccessor( oldDesc.asPointer(), oldDesc.getClassMember().getStructClass(), "stride").asInt32();
		
		hkClassMemberAccessor oldDecls( oldDesc.asPointer(), oldDesc.getClassMember().getStructClass(), "decls");
		hkClassMemberAccessor newDecls( newDesc.asRaw(), newDesc.getClassMember().getStructClass(), "decls");

		const hkClass& oldElemsClass = oldDecls.getClassMember().getStructClass();
		const hkClass& newElemsClass = newDecls.getClassMember().getStructClass();

		const hkClassMemberAccessor::SimpleArray& oldElems = oldDecls.asSimpleArray();
		initArray( newDecls, oldElems.size, tracker );
		hkClassMemberAccessor::SimpleArray& newElems = newDecls.asSimpleArray();

		hkClassMemberAccessor oldData( oldObj, "vertexData");
		hkClassMemberAccessor newData( newObj, "data");
		const hkClassMemberAccessor::HomogeneousArray& oldBuf = oldData.asHomogeneousArray();
		int numVertices = oldBuf.size;
		newData.member("numVerts").asInt32() = numVertices;

		hkArray<hkVector4>& vectorData = *static_cast<hkArray<hkVector4>*>( newData.member("vectorData").asRaw() );
		hkArray<hkUint8>& uint8Data = *static_cast<hkArray<hkUint8>*>( newData.member("uint8Data").asRaw() );
		hkArray<hkUint16>& uint16Data = *static_cast<hkArray<hkUint16>*>( newData.member("uint16Data").asRaw() );
		hkArray<hkUint32>& uint32Data = *static_cast<hkArray<hkUint32>*>( newData.member("uint32Data").asRaw() );
		hkArray<float>& floatData = *static_cast<hkArray<float>*>( newData.member("floatData").asRaw() );
		
		for( int ei = 0; ei < oldElems.size; ++ei )
		{
			enum DataType
			{
				HKX_DT_NONE = 0,
				HKX_DT_UINT8, // only used for four contiguous hkUint8s, hkUint8[4]
				HKX_DT_INT16, // only used for old style quantized tcoords (0x7fff maps to 10.0f), so div by 3276.7f to get the float tcoords. Deprecated.
				HKX_DT_UINT32,
				HKX_DT_FLOAT,
				HKX_DT_FLOAT2, // for tex coords 
				HKX_DT_FLOAT3, // will always be 16byte aligned, so you can treat as a hkVector4 (with undefined w, with SIMD enabled etc)
				HKX_DT_FLOAT4  // will always be 16byte aligned, so you can treat as a hkVector4 (with SIMD enabled etc)
			};
			enum DataUsage
			{
				HKX_DU_NONE = 0,
				HKX_DU_POSITION = 1,
				HKX_DU_COLOR = 2,    // first color always can be assumed to be per vertex Diffuse, then per vertex Specular (rare)
				HKX_DU_NORMAL = 4,
				HKX_DU_TANGENT = 8,
				HKX_DU_BINORMAL = 16, // aka BITANGENT
				HKX_DU_TEXCOORD = 32, // in order, 0,1,2, etc of the texture channels. Assumed to be 2D, [u,v], in most cases
				HKX_DU_BLENDWEIGHTS = 64,  // usually 4 weights, but 3 can be stored with 1 implied. Can be stored as 4*uint8, so quantized where 1.0f => 0xff (255),
				HKX_DU_BLENDINDICES = 128, // usually 4 hkUint8s in a row. So can reference 256 blend transforms (bones)
				HKX_DU_USERDATA = 256
			};

			hkClassAccessor od( hkAddByteOffset(oldElems.data, ei * oldElemsClass.getObjectSize()), &oldElemsClass);
			hkClassAccessor nd( hkAddByteOffset(newElems.data, ei * newElemsClass.getObjectSize()), &newElemsClass);

			nd.member("type").asInt16() = od.member("type").asInt16();
			nd.member("usage").asInt16() = od.member("usage").asInt16();

			if( nd.member("usage").asInt16() == HKX_DU_TEXCOORD && nd.member("type").asInt16() == HKX_DT_FLOAT )
			{
				nd.member("type").asInt16() = HKX_DT_FLOAT2;
			}
			switch( nd.member("type").asInt16() )
			{
				case HKX_DT_UINT8:
				{
					nd.member("byteOffset").asInt32() = uint8Data.getSize(); // new offset from start of array
					nd.member("byteStride").asInt32() = 4;
					int offset = od.member("byteOffset").asInt32(); // old offset from start of vertex

					hkUint8* dst = uint8Data.expandBy(numVertices*4);
					for( int i = 0; i < numVertices; ++i )
					{
						hkUint8* src = (hkUint8*)hkAddByteOffset( oldBuf.data, i * oldVertexStride + offset);
						dst[i*4+0] = src[0];
						dst[i*4+1] = src[1];
						dst[i*4+2] = src[2];
						dst[i*4+3] = src[3];
					}
					break;
				}
				case HKX_DT_INT16: // only used for old style quantized tcoords (0x7fff maps to 10.0f), so div by 3276.7f to get the float tcoords. Deprecated.
				{
					nd.member("byteOffset").asInt32() = uint16Data.getSize()*sizeof(hkInt16); // new offset from start of array
					nd.member("byteStride").asInt32() = 2*sizeof(hkInt16);
					int offset = od.member("byteOffset").asInt32(); // old offset from start of vertex

					hkUint16* dst = uint16Data.expandBy(numVertices*2);
					for( int i = 0; i < numVertices; ++i )
					{
						hkUint16* src = (hkUint16*)hkAddByteOffset( oldBuf.data, i * oldVertexStride + offset);
						dst[i*2+0] = src[0];
						dst[i*2+1] = src[1];
					}
					break;
				}
				case HKX_DT_UINT32: // vertex color normally
				{
					nd.member("byteOffset").asInt32() = uint32Data.getSize()*sizeof(hkUint32); // new offset from start of array
					nd.member("byteStride").asInt32() = sizeof(hkInt32);
					int offset = od.member("byteOffset").asInt32(); // old offset from start of vertex

					hkUint32* dst = uint32Data.expandBy(numVertices);
					for( int i = 0; i < numVertices; ++i )
					{
						dst[i] = *(hkUint32*)hkAddByteOffset( oldBuf.data, i * oldVertexStride + offset);
					}
					break;
				}
				case HKX_DT_FLOAT:
				{
					nd.member("byteOffset").asInt32() = floatData.getSize() * sizeof(hkReal); // new offset from start of array
					nd.member("byteStride").asInt32() = sizeof(hkReal);
					int offset = od.member("byteOffset").asInt32(); // old offset from start of vertex

					float* vp = floatData.expandBy(numVertices);
					for( int i = 0; i < numVertices; ++i )
					{
						vp[i] = *(float*)hkAddByteOffset( oldBuf.data, i * oldVertexStride + offset );
					}
					break;
				}

				case HKX_DT_FLOAT2:
				{
					nd.member("byteOffset").asInt32() = floatData.getSize() * 2 * sizeof(hkReal); // new offset from start of array
					nd.member("byteStride").asInt32() = 2 * sizeof(hkReal);
					int offset = od.member("byteOffset").asInt32(); // old offset from start of vertex

					float* vp = floatData.expandBy(2*numVertices);
					for( int i = 0; i < numVertices; ++i )
					{
						float* src = (float*)hkAddByteOffset( oldBuf.data, i * oldVertexStride + offset );
						vp[2*i+0] = src[0];
						vp[2*i+1] = src[1];
					}
					break;
				}
				case HKX_DT_FLOAT3: // will always be 16byte aligned, so you can treat as a hkVector4 (with undefined w, with SIMD enabled etc)
				case HKX_DT_FLOAT4:
				{
					nd.member("byteOffset").asInt32() = vectorData.getSize() * sizeof(hkVector4); // new offset from start of array
					nd.member("byteStride").asInt32() = sizeof(hkVector4);
					int offset = od.member("byteOffset").asInt32(); // old offset from start of vertex

					hkVector4* vp = vectorData.expandBy(numVertices);
					for( int i = 0; i < numVertices; ++i )
					{
						vp[i] = *(hkVector4*)hkAddByteOffset( oldBuf.data, i * oldVertexStride + offset);
					}
					break;
				}
				default:
				{
					HK_ASSERT(0x2c284b4f, 0);
				}
			}
		}
		// register new array allocations with tracker
		registerNewArray(reinterpret_cast<DummyArray&>(vectorData), hkSizeOf(hkVector4), tracker);
		registerNewArray(reinterpret_cast<DummyArray&>(uint8Data), hkSizeOf(hkUint8), tracker);
		registerNewArray(reinterpret_cast<DummyArray&>(uint16Data), hkSizeOf(hkUint16), tracker);
		registerNewArray(reinterpret_cast<DummyArray&>(uint32Data), hkSizeOf(hkUint32), tracker);
		registerNewArray(reinterpret_cast<DummyArray&>(floatData), hkSizeOf(float), tracker);
	}

	static void Update_hkxVertexDescription( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		//HK_ASSERT(0x13ff6aaa, false); // TODO, copy old data to new data structures
	}

	static void Update_hkxVertexDescriptionElementDecl( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		HK_ASSERT(0x13ff6aaa, false); // TODO, copy old data to new data structures
	}

	static inline void CheckAndUpdate_VariantCompatibility( const hkVariant& oldVariant, const hkClassMemberAccessor::Pointer& newVariant, hkObjectUpdateTracker& tracker )
	{
		if( newVariant == HK_NULL )
		{
			// nothing to check
			return;
		}
		#ifdef HK_DEBUG
			if( oldVariant.m_class )
			{
				const hkClass* newReferencedObjectClass = Context::s_context->findNewClassFromOld(HK_COMPAT_VERSION_FROM::hkReferencedObjectClass);
				HK_ASSERT(0x1f617876, newReferencedObjectClass);
				const hkClass* newClass = Context::s_context->findNewClassFromOld(*oldVariant.m_class);
				HK_ASSERT(0x5a17951c, newClass);
				HK_ASSERT3(0x6eb83899, newReferencedObjectClass->isSuperClass(*newClass), "Found hkRefVariant containing non-reference counted object 0x" << newVariant << ".\n"
						"The '" << oldVariant.m_class->getName() << "' class must be reference counted to load the variant in version " << HK_COMPAT_VERSION_TO::VersionString << ".");
				//tracker.replaceObject(newVariant, HK_NULL, HK_NULL);
			}
		#endif
	}
	static inline void CheckAndUpdate_VariantMember( const hkClassMemberAccessor& oldVariant, const hkClassMemberAccessor& newVariant, hkObjectUpdateTracker& tracker )
	{
		CheckAndUpdate_VariantCompatibility(oldVariant.asVariant(), newVariant.asPointer(), tracker);
	}

	static inline void CheckAndUpdate_ArrayOfVariantsMember( const hkClassMemberAccessor& oldVariantArray, const hkClassMemberAccessor& newVariantArray, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor::SimpleArray& oldArray = oldVariantArray.asSimpleArray();
		hkClassMemberAccessor::SimpleArray& newArray = newVariantArray.asSimpleArray();
		HK_ASSERT(0x3b699373, oldArray.size == newArray.size);
		for( int i = 0; i < oldArray.size; ++i )
		{
			const hkVariant& oldVariant = static_cast<hkVariant*>(oldArray.data)[i];
			const hkClassMemberAccessor::Pointer& newVariant = static_cast<hkClassMemberAccessor::Pointer*>(newArray.data)[i];
			CheckAndUpdate_VariantCompatibility(oldVariant, newVariant, tracker);
		}
	}

	static inline void Update_ObjectArrayMember( const hkClassMemberAccessor& oldArray, const hkClassMemberAccessor& newArray, hkVersionRegistry::VersionFunc verFunc, hkObjectUpdateTracker& tracker )
	{
		const hkClass* oldClass = oldArray.getClassMember().getClass();
		HK_ASSERT(0x31001a77, oldClass);
		const hkClass* newClass = newArray.getClassMember().getClass();
		HK_ASSERT(0x7979bdae, newClass);
		hkClassMemberAccessor::SimpleArray& oldObjectArray = oldArray.asSimpleArray();
		hkClassMemberAccessor::SimpleArray& newObjectArray = newArray.asSimpleArray();
		HK_ASSERT(0x5b2bb460, oldObjectArray.size == newObjectArray.size);
		for( int i = 0; i < oldObjectArray.size; ++i )
		{
			hkVariant oldObject = { hkAddByteOffset(oldObjectArray.data, i*oldClass->getObjectSize()), oldClass };
			hkVariant newObject = { hkAddByteOffset(newObjectArray.data, i*newClass->getObjectSize()), newClass };
			(*verFunc)(oldObject, newObject, tracker);
		}
	}

	static void Update_hkaBoneAttachment( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		CheckAndUpdate_VariantMember(hkClassMemberAccessor(oldObj, "attachment"), hkClassMemberAccessor(newObj, "attachment"), tracker);
	}

	static void Update_hkxAttribute( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		CheckAndUpdate_VariantMember(hkClassMemberAccessor(oldObj, "value"), hkClassMemberAccessor(newObj, "value"), tracker);
	}

	static void Update_hkxAttributeGroup( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		Update_ObjectArrayMember(hkClassMemberAccessor(oldObj, "attributes"), hkClassMemberAccessor(newObj, "attributes"), Update_hkxAttribute, tracker);
	}

	static void Update_hkdShape( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		Update_ObjectArrayMember(hkClassMemberAccessor(oldObj, "attributes"), hkClassMemberAccessor(newObj, "attributes"), Update_hkxAttribute, tracker);
	}

	static void Update_hkdBody( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		Update_ObjectArrayMember(hkClassMemberAccessor(oldObj, "attributes"), hkClassMemberAccessor(newObj, "attributes"), Update_hkxAttribute, tracker);
	}

	static void Update_hkxAttributeHolder( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		Update_ObjectArrayMember(hkClassMemberAccessor(oldObj, "attributeGroups"), hkClassMemberAccessor(newObj, "attributeGroups"), Update_hkxAttributeGroup, tracker);
	}

	static void Update_hkxNodeSelectionSet( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		Update_hkxAttributeHolder(oldObj, newObj, tracker);
	}

	static void Update_hkxNode( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		Update_hkxAttributeHolder(oldObj, newObj, tracker);
		CheckAndUpdate_VariantMember(hkClassMemberAccessor(oldObj, "object"), hkClassMemberAccessor(newObj, "object"), tracker);
	}

	static void Update_hkxMaterialTextureStage( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		CheckAndUpdate_VariantMember(hkClassMemberAccessor(oldObj, "texture"), hkClassMemberAccessor(newObj, "texture"), tracker);
	}

	static void Update_hkxMaterial( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		Update_hkxAttributeHolder(oldObj, newObj, tracker);
		Update_ObjectArrayMember(hkClassMemberAccessor(oldObj, "stages"), hkClassMemberAccessor(newObj, "stages"), Update_hkxMaterialTextureStage, tracker);
		CheckAndUpdate_VariantMember(hkClassMemberAccessor(oldObj, "extraData"), hkClassMemberAccessor(newObj, "extraData"), tracker);
	}

	static void Update_hkxMeshSection( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		CheckAndUpdate_ArrayOfVariantsMember(hkClassMemberAccessor(oldObj, "userChannels"), hkClassMemberAccessor(newObj, "userChannels"), tracker);
	}

	static void Update_hkxMeshUserChannelInfo( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		Update_hkxAttributeHolder(oldObj, newObj, tracker);
	}

	static void Update_hkxMesh( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldUserChannelInfos(oldObj, "userChannelInfos");
		hkClassMemberAccessor newUserChannelInfos(newObj, "userChannelInfos");

		// version hkxMeshUserChannelInfo objects manually
		hkArray<hkVariant> newUserChannelInfoVariants;
		{
			hkClassMemberAccessor::SimpleArray& oldObjectArray = oldUserChannelInfos.asSimpleArray();
			if( oldObjectArray.size > 0 )
			{
				hkArray<hkVariant> oldUserChannelInfoVariants;
				oldUserChannelInfoVariants.setSize(oldObjectArray.size);
				newUserChannelInfoVariants.setSize(oldObjectArray.size);
				const hkClass* oldUserChannelInfoClass = oldUserChannelInfos.getClassMember().getClass();
				HK_ASSERT(0x18bff4df, oldUserChannelInfoClass);
				int oldUserChannelInfoSize = oldUserChannelInfoClass->getObjectSize();
				const hkClass* newUserChannelInfoClass = newUserChannelInfos.getClassMember().getClass();
				HK_ASSERT(0x22ef49c0, newUserChannelInfoClass);
				for( int i = 0; i < oldObjectArray.size; ++i )
				{
					oldUserChannelInfoVariants[i].m_class = oldUserChannelInfoClass;
					oldUserChannelInfoVariants[i].m_object = hkAddByteOffset(oldObjectArray.data, i*oldUserChannelInfoSize);

					newUserChannelInfoVariants[i].m_class = newUserChannelInfoClass;
					newUserChannelInfoVariants[i].m_object = HK_NULL;
				}
				HK_ON_DEBUG(void* allocations =)hkVersionUtil::copyObjects(oldUserChannelInfoVariants, newUserChannelInfoVariants, tracker);
				HK_ASSERT(0x30e03711, allocations);
				for( int i = 0; i < oldObjectArray.size; ++i )
				{
					Context::s_context->registerNewObjectForVersioning(newUserChannelInfoVariants[i]);
					Update_hkxMeshUserChannelInfo(oldUserChannelInfoVariants[i], newUserChannelInfoVariants[i], tracker);
				}
			}
		}
		
		// setup new hkxMesh::m_userChannelInfos array of pointers
		{
			initArray(newUserChannelInfos, newUserChannelInfoVariants.getSize(), tracker);
			hkClassMemberAccessor::SimpleArray& newPointerArray = newUserChannelInfos.asSimpleArray();
			for( int i = 0; i < newUserChannelInfoVariants.getSize(); ++i )
			{
				tracker.objectPointedBy(newUserChannelInfoVariants[i].m_object, hkAddByteOffset(newPointerArray.data, i*sizeof(void*)));
			}
		}
	}

	static void Update_hkRootLevelContainerNamedVariant( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		CheckAndUpdate_VariantMember(hkClassMemberAccessor(oldObj, "variant"), hkClassMemberAccessor(newObj, "variant"), tracker);
	}

	static void Update_hkRootLevelContainer( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		Update_ObjectArrayMember(hkClassMemberAccessor(oldObj, "namedVariants"), hkClassMemberAccessor(newObj, "namedVariants"), Update_hkRootLevelContainerNamedVariant, tracker);
	}

	static void Update_hkMemoryResourceHandle( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		CheckAndUpdate_VariantMember(hkClassMemberAccessor(oldObj, "variant"), hkClassMemberAccessor(newObj, "variant"), tracker);
	}

static const hkVersionRegistry::ClassAction s_updateActions[] =
{
	// variants
	{ 0x1388d601, 0x1388d601, hkVersionRegistry::VERSION_VARIANT, "hkCustomAttributesAttribute", HK_NULL },
	{ 0xbff19005, 0xbff19005, hkVersionRegistry::VERSION_VARIANT, "hkCustomAttributes", HK_NULL },
	{ 0x15d99dc6, 0x15d99dc6, hkVersionRegistry::VERSION_VARIANT, "hkbVariableValueSet", HK_NULL },

	// Common
	{ 0x72e8e849, 0xf2edcc5f, hkVersionRegistry::VERSION_COPY, "hkxMesh", Update_hkxMesh },	// hkReferencedObject, member types changed, compatible
	{ 0x64e9a03c, 0x270724a5, hkVersionRegistry::VERSION_COPY, "hkxMeshUserChannelInfo", Update_hkxMeshUserChannelInfo },	// hkReferencedObject, member types changed, compatible
	{ 0x914da6c1, 0x7375cae3, hkVersionRegistry::VERSION_COPY, "hkxAttribute", Update_hkxAttribute }, // COM-629
	{ 0x1667c01c, 0x345ca95d, hkVersionRegistry::VERSION_COPY, "hkxAttributeGroup", Update_hkxAttributeGroup },	// COM-629, hkReferencedObject, member types changed, compatible
	{ 0xbe6765dd, 0x02a30cda, hkVersionRegistry::VERSION_COPY, "hkxMaterial", Update_hkxMaterial },	// hkReferencedObject, member types changed, compatible
	{ 0x02ea23f0, 0xfa6facb2, hkVersionRegistry::VERSION_COPY, "hkxMaterialTextureStage", Update_hkxMaterialTextureStage },	// COM-629, member types changed, compatible
	{ 0xf598a34e, 0x2772c11e, hkVersionRegistry::VERSION_COPY, "hkRootLevelContainer", Update_hkRootLevelContainer }, // namedVariant is hkArray, 700-$$
	{ 0x853a899c, 0xb103a2cd, hkVersionRegistry::VERSION_COPY, "hkRootLevelContainerNamedVariant", Update_hkRootLevelContainerNamedVariant }, // hkVariant is hkRefVariant
	BINARY_IDENTICAL(0x882d9a6e, 0xa6815115, "hkxEnvironmentVariable"),						// COM-629, 700-$$
	{ 0x1adf8fa3, 0x041e1aa5, hkVersionRegistry::VERSION_COPY, "hkxEnvironment", HK_NULL }, // COM-629, 700-$$
	{ 0xe517ce6a, 0x0207cb01, hkVersionRegistry::VERSION_COPY, "hkAlignSceneToNodeOptions", Update_hkAlignSceneToNodeOptions },	// m_nodeName type changed
	{ 0x9bb15af4, 0xce8b2fbd, hkVersionRegistry::VERSION_COPY, "hkxAnimatedFloat", HK_NULL },	// m_floats type changed, compatible
	{ 0x95bd90ad, 0x5838e337, hkVersionRegistry::VERSION_COPY, "hkxAnimatedMatrix", HK_NULL },	// m_matrices type changed, compatible
	{ 0x9bb6e38a, 0xb4f01baa, hkVersionRegistry::VERSION_COPY, "hkxAnimatedQuaternion", HK_NULL },	// m_quaternions type changed, compatible
	{ 0xfe98cabd, 0x34b1a197, hkVersionRegistry::VERSION_COPY, "hkxAnimatedVector", HK_NULL },	// m_vectors type changed, compatible
	{ 0xd5c65fae, 0xe3597b02, hkVersionRegistry::VERSION_COPY, "hkxCamera", HK_NULL },	// hkReferencedObject, compatible
	{ 0x2cebe88e, 0x9ad32a5e, hkVersionRegistry::VERSION_COPY, "hkxEdgeSelectionChannel", HK_NULL },	// hkReferencedObject, m_selectedEdges type changed, compatible
	{ 0x1c8a8c37, 0xc12c8197, hkVersionRegistry::VERSION_COPY, "hkxIndexBuffer", HK_NULL },	// hkReferencedObject, m_indices16/m_indices32 type changed, compatible
	{ 0x8e92a993, 0x81c86d42, hkVersionRegistry::VERSION_COPY, "hkxLight", HK_NULL },	// hkReferencedObject, compatible
	{ 0xf6a7f6f9, 0x1d39f925, hkVersionRegistry::VERSION_COPY, "hkxMaterialEffect", HK_NULL },	// hkReferencedObject, m_name/m_data type changed, compatible
	{ 0xa8d8b05c, 0x28515eff, hkVersionRegistry::VERSION_COPY, "hkxMaterialShader", HK_NULL },	// hkReferencedObject, member types changed, compatible
	{ 0x2011dc1c, 0x154650f3, hkVersionRegistry::VERSION_COPY, "hkxMaterialShaderSet", HK_NULL },	// hkReferencedObject, member types changed, compatible
	BINARY_IDENTICAL(0x521e9517, 0x433dee92, "hkxNodeAnnotationData"),	// member types changed, compatible
	{ 0x912c8863, 0xe2286cf8, hkVersionRegistry::VERSION_COPY, "hkxMeshSection", Update_hkxMeshSection },	// COM-629, hkReferencedObject, member types changed, compatible
	{ 0x06af1b5a, 0x5a218502, hkVersionRegistry::VERSION_COPY, "hkxNode", Update_hkxNode },	// COM-629, hkReferencedObject, member types changed, compatible
	{ 0x158bea87, 0xd753fc4d, hkVersionRegistry::VERSION_COPY, "hkxNodeSelectionSet", Update_hkxNodeSelectionSet },	// hkReferencedObject, member types changed, compatible
	{ 0x445a443a, 0x7468cc44, hkVersionRegistry::VERSION_COPY, "hkxAttributeHolder", Update_hkxAttributeHolder },	// hkReferencedObject, member types changed, compatible
	{ 0x1fb22361, 0x5f673ddd, hkVersionRegistry::VERSION_COPY, "hkxScene", HK_NULL },	// hkReferencedObject, member types changed, compatible
	{ 0xc532c710, 0x09b9d9a9, hkVersionRegistry::VERSION_COPY, "hkxSkinBinding", HK_NULL },	// hkReferencedObject, member types changed, compatible
	{ 0x901601e1, 0x7a894596, hkVersionRegistry::VERSION_COPY, "hkxSparselyAnimatedBool", HK_NULL },	// hkReferencedObject, member types changed, compatible
	{ 0x045fc0ef, 0x68a47b64, hkVersionRegistry::VERSION_COPY, "hkxSparselyAnimatedEnum", HK_NULL },	// hkReferencedObject, member types changed, compatible
	{ 0x4c0e8f4a, 0xca961951, hkVersionRegistry::VERSION_COPY, "hkxSparselyAnimatedInt", HK_NULL },	// hkReferencedObject, member types changed, compatible
	{ 0xd907cff8, 0x185da6fd, hkVersionRegistry::VERSION_COPY, "hkxSparselyAnimatedString", Update_hkxSparselyAnimatedString },	// hkReferencedObject, member types changed
	REMOVED("hkxSparselyAnimatedStringStringType"),
	{ 0x548ff417, 0x1e289259, hkVersionRegistry::VERSION_COPY, "hkxTextureFile", HK_NULL },	// hkReferencedObject, member types changed, compatible
	{ 0x76dfe21d, 0xd45841d6, hkVersionRegistry::VERSION_COPY, "hkxTextureInplace", HK_NULL },	// hkReferencedObject, member types changed, compatible
	{ 0xe50fba23, 0xa02cfca9, hkVersionRegistry::VERSION_COPY, "hkxTriangleSelectionChannel", HK_NULL },	// hkReferencedObject, member types changed, compatible
	{ 0x8e9b1727, 0x20d9d796, hkVersionRegistry::VERSION_COPY, "hkxVertexBuffer", Update_hkxVertexBuffer }, // removed homogeneous array, member types changed
	{ 0xe745584c, 0x305e43cd, hkVersionRegistry::VERSION_COPY, "hkxVertexDescription", Update_hkxVertexDescription },	// member types changed
	{ 0x402a5349, 0x0c076c69, hkVersionRegistry::VERSION_COPY, "hkxVertexDescriptionElementDecl", Update_hkxVertexDescriptionElementDecl },	// member types changed
	{ 0x49557cc2, 0xbeeb397c, hkVersionRegistry::VERSION_COPY, "hkxVertexFloatDataChannel", HK_NULL },	// hkReferencedObject, member types changed, compatible
	{ 0x0c0df26b, 0x5a50e673, hkVersionRegistry::VERSION_COPY, "hkxVertexIntDataChannel", HK_NULL },	// hkReferencedObject, member types changed, compatible
	{ 0x7d0768c4, 0x866ec6d0, hkVersionRegistry::VERSION_COPY, "hkxVertexSelectionChannel", HK_NULL },	// hkReferencedObject, member types changed, compatible
	{ 0x22874dab, 0x2ea63179, hkVersionRegistry::VERSION_COPY, "hkxVertexVectorDataChannel", HK_NULL },	// hkReferencedObject, member types changed, compatible
	{ 0x93561016, 0x4762f92a, hkVersionRegistry::VERSION_COPY, "hkMemoryResourceContainer", HK_NULL },	// member types changed, member removed, compatible
	BINARY_IDENTICAL(0x3d55b593, 0x2c76ce16, "hkMonitorStreamStringMapStringMap"),	// member types changed, compatible
	BINARY_IDENTICAL(0x858de884, 0xc4d3a8b4, "hkMonitorStreamStringMap"),	// hkMonitorStreamStringMapStringMap changed
	BINARY_IDENTICAL(0xa189b867, 0x738fca05, "hkMonitorStreamColorTableColorPair"),	// member types changed, compatible
	BINARY_IDENTICAL(0xc1b73a67, 0x79e53e85, "hkMonitorStreamColorTable"),	// hkMonitorStreamColorTableColorPair changed
	{ 0xdce3ca6b, 0xbffac086, hkVersionRegistry::VERSION_COPY, "hkMemoryResourceHandle", Update_hkMemoryResourceHandle }, // COM-629, member types changed, removed members
	{ 0xf1d32622, 0x3144d17c, hkVersionRegistry::VERSION_COPY, "hkMemoryResourceHandleExternalLink", HK_NULL },	// member types changed, removed members
	BINARY_IDENTICAL(0x3b25af2a, 0x6075f3ff, "hkMeshSectionCinfo"),	// member +nosave, compatible
	BINARY_IDENTICAL(0x373988ae, 0xb743a578, "hkMemoryMeshShape"),	// COM-629 + hkMeshSectionCinfo changed
	BINARY_IDENTICAL(0x4b2533f6, 0x11545121, "hkSimpleLocalFrame"),	// COM-629
	BINARY_IDENTICAL(0xbac80827, 0xb1a96c2f, "hkLocalFrameGroup"),	// COM-629
	BINARY_IDENTICAL(0x41397b81, 0x1893c365, "hkMeshSection"),	// +nosave member
	{ 0x6b6814de, 0xc8ae86a7, hkVersionRegistry::VERSION_COPY, "hkpPhysicsSystemDisplayBinding", HK_NULL },
	{ 0x466e12ed, 0x2b95f996, hkVersionRegistry::VERSION_COPY, "hkpRigidBodyDisplayBinding", HK_NULL },
	{ 0x2c40dbe5, 0xdc46c906, hkVersionRegistry::VERSION_COPY, "hkpDisplayBindingData", HK_NULL },

	REMOVED("hkxVertexP4N4C1T10"),
	REMOVED("hkxVertexP4N4C1T2"),
	REMOVED("hkxVertexP4N4C1T6"),
	REMOVED("hkxVertexP4N4T4"),
	REMOVED("hkxVertexP4N4T4B4C1T10"),
	REMOVED("hkxVertexP4N4T4B4C1T2"),
	REMOVED("hkxVertexP4N4T4B4C1T6"),
	REMOVED("hkxVertexP4N4T4B4T4"),
	REMOVED("hkxVertexP4N4T4B4W4I4C1Q2"),
	REMOVED("hkxVertexP4N4T4B4W4I4C1T12"),
	REMOVED("hkxVertexP4N4T4B4W4I4C1T4"),
	REMOVED("hkxVertexP4N4T4B4W4I4C1T8"),
	REMOVED("hkxVertexP4N4T4B4W4I4Q4"),
	REMOVED("hkxVertexP4N4T4B4W4I4T6"),
	REMOVED("hkxVertexP4N4W4I4C1Q2"),
	REMOVED("hkxVertexP4N4W4I4C1T12"),
	REMOVED("hkxVertexP4N4W4I4C1T4"),
	REMOVED("hkxVertexP4N4W4I4C1T8"),
	REMOVED("hkxVertexP4N4W4I4T2"),
	REMOVED("hkxVertexP4N4W4I4T6"),

	// Destruction
	{ 0xd2963e7c, 0xf307ff62, hkVersionRegistry::VERSION_COPY, "hkdShape", Update_hkdShape },	// COM-629
	{ 0x1bbfdb97, 0xb30237d9, hkVersionRegistry::VERSION_COPY, "hkdBody", Update_hkdBody },		// COM-629
	{ 0xe4ed1a08, 0xb03555fd, hkVersionRegistry::VERSION_COPY, "hkdVoronoiFracture", HK_NULL },	// HKD-343
	{ 0xfe812f2d, 0xf5fb969,  hkVersionRegistry::VERSION_COPY, "hkdDestructionDemoConfig", HK_NULL },		// COM-629
	{ 0xbc0b22cf, 0xd3921ff7, hkVersionRegistry::VERSION_COPY, "hkdBreakableShapeContactArea", HK_NULL },	// COM-629
	{ 0x55ca2d3c, 0x76658b24, hkVersionRegistry::VERSION_COPY, "hkdBreakableShape", HK_NULL },				// COM-629
	{ 0x9b383de9, 0xff03af3a, hkVersionRegistry::VERSION_COPY, "hkdRaycastGun", HK_NULL },					// 700-$$
	{ 0x5bdd6792, 0xc89c7f72, hkVersionRegistry::VERSION_COPY, "hkdController", HK_NULL },					// 700-$$
	{ 0x7d4c7b72, 0xaa3e9892, hkVersionRegistry::VERSION_COPY, "hkdAction", HK_NULL },						// 700-$$
	BINARY_IDENTICAL(0xeb2364c1, 0xee3c2aec, "hkdBreakableBodySmallArraySerializeOverrideType"),			// 700-$$
	BINARY_IDENTICAL(0xf6d23796, 0xf1105a48, "hkdBreakableBody"),											// 700-$$
	BINARY_IDENTICAL(0x7d37d646, 0x6c14adc3, "hkdGeometryObjectIdentifier"),								// COM-629
	BINARY_IDENTICAL(0x137456e3, 0x335c4983, "hkdGeometry"),												// COM-629
	{ 0xd90d28a9, 0x980dc9b8, hkVersionRegistry::VERSION_COPY, "hkdFracture", HK_NULL }, // removed deprecated m_connectivityType

	// Physics
	// New member added, padding
	{ 0x695c1674, 0xb8168996, hkVersionRegistry::VERSION_COPY, "hkpExtendedMeshShape", HK_NULL },			// 700-$$
	// Added new CollectionType values.
	BINARY_IDENTICAL(0x208eee42, 0xe8c3991d, "hkpShapeCollection"),											// 700-$$
	{ 0xe7f760b1, 0x2f10b493, hkVersionRegistry::VERSION_COPY, "hkpEntity", Update_hkpEntity },				// 700-$$
	BINARY_IDENTICAL(0x4c105f22, 0xf557023c, "hkpEntityExtendedListeners"),									// 700-$$
	BINARY_IDENTICAL(0xeb2364c1, 0xee3c2aec, "hkpEntitySmallArraySerializeOverrideType"),					// 700-$$
	BINARY_IDENTICAL(0xeb2364c1, 0xee3c2aec, "hkpConstraintInstanceSmallArraySerializeOverrideType"),		// 700-$$
	BINARY_IDENTICAL(0x418c7656, 0x06de8f8e, "hkpConstraintInstance"),										// 700-$$
	BINARY_IDENTICAL(0x44fddf96, 0x852ab70b, "hkpFirstPersonGun"),											// COM-629
	BINARY_IDENTICAL(0xce5e4f30, 0x4e32287c, "hkContactPointMaterial"),										// 700-$$
	BINARY_IDENTICAL(0xd33c512e, 0x878cbf81, "hkpConstraintAtom"),											// 700-$$
	BINARY_IDENTICAL(0x55c24b79, 0x65cd952a, "hkpBridgeAtoms"),												// 700-$$
	BINARY_IDENTICAL(0xfbcb20b8, 0x4feeb8e2, "hkpBallAndSocketConstraintData"),								// 700-$$
	BINARY_IDENTICAL(0x3e208629, 0x899efbb9, "hkpBallAndSocketConstraintDataAtoms"),						// 700-$$
	BINARY_IDENTICAL(0x4b9c419, 0xab70d590, "hkpBallSocketChainData"),										// 700-$$
	BINARY_IDENTICAL(0x6f0be4b1, 0xdf812dd4, "hkpBreakableConstraintData"),									// 700-$$
	BINARY_IDENTICAL(0x9800d734, 0x9f23b5de, "hkpGenericConstraintData"),									// 700-$$
	BINARY_IDENTICAL(0xa8ef77c, 0x14ffc72d, "hkpHingeConstraintData"),										// 700-$$
	BINARY_IDENTICAL(0xd9d0ebf0, 0x66a6fb1c, "hkpHingeConstraintDataAtoms"),								// 700-$$
	BINARY_IDENTICAL(0x406417ae, 0x9c7de556, "hkpHingeLimitsData"),											// 700-$$
	BINARY_IDENTICAL(0x47e00ae1, 0x3b5747b1, "hkpHingeLimitsDataAtoms"),									// 700-$$
	BINARY_IDENTICAL(0xd458bb24, 0x1c566b11, "hkpLimitedHingeConstraintData"),								// 700-$$
	BINARY_IDENTICAL(0x207b4cda, 0x37649b2e, "hkpLimitedHingeConstraintDataAtoms"),							// 700-$$
	BINARY_IDENTICAL(0xa4bf9c94, 0x5ac85b1a, "hkpMalleableConstraintData"),									// 700-$$
	BINARY_IDENTICAL(0xca168aae, 0x6db0556, "hkpPointToPathConstraintData"),								// 700-$$
	BINARY_IDENTICAL(0x49757895, 0x44e0078f, "hkpPointToPlaneConstraintData"),								// 700-$$
	BINARY_IDENTICAL(0x3a0aa8e, 0x105a62a0, "hkpPointToPlaneConstraintDataAtoms"),							// 700-$$
	BINARY_IDENTICAL(0x4a138d43, 0xa8ad141f, "hkpPoweredChainData"),										// 700-$$
	BINARY_IDENTICAL(0xbff614e0, 0x9a63ebb8, "hkpPrismaticConstraintData"),									// 700-$$
	BINARY_IDENTICAL(0xa19b09b6, 0x918f0da0, "hkpPrismaticConstraintDataAtoms"),							// 700-$$
	BINARY_IDENTICAL(0x130b2d6, 0xb5032c92, "hkpPulleyConstraintData"),										// 700-$$
	BINARY_IDENTICAL(0xecd4b4be, 0xbfa59ef6, "hkpPulleyConstraintDataAtoms"),								// 700-$$
	BINARY_IDENTICAL(0x43c94a7f, 0x17a4d42f, "hkpRagdollConstraintData"),									// 700-$$
	BINARY_IDENTICAL(0x407bf11b, 0x52dc039e, "hkpRagdollConstraintDataAtoms"),								// 700-$$
	BINARY_IDENTICAL(0x9f00917f, 0x26047f53, "hkpRagdollLimitsData"),										// 700-$$
	BINARY_IDENTICAL(0x93990770, 0xcebae0e0, "hkpRagdollLimitsDataAtoms"),									// 700-$$
	BINARY_IDENTICAL(0x93d5253c, 0x187227c8, "hkpRotationalConstraintData"),								// 700-$$
	BINARY_IDENTICAL(0xd6b4fa0e, 0xf1dd4c8d, "hkpRotationalConstraintDataAtoms"),							// 700-$$
	BINARY_IDENTICAL(0xf8bf3859, 0x5a166a0, "hkpSerializedAgentNnEntry"),									// 700-$$
	BINARY_IDENTICAL(0xef7de5b0, 0x3066daaa, "hkpStiffSpringChainData"),									// 700-$$
	BINARY_IDENTICAL(0x27f20ebe, 0x422788c8, "hkpStiffSpringConstraintData"),								// 700-$$
	BINARY_IDENTICAL(0x6177df03, 0x13aa4117, "hkpStiffSpringConstraintDataAtoms"),							// 700-$$
	BINARY_IDENTICAL(0x3dff463d, 0x81bb69a0, "hkpWheelConstraintData"),										// 700-$$
	BINARY_IDENTICAL(0xbe9b5f8b, 0x81bf4bb6, "hkpWheelConstraintDataAtoms"),								// 700-$$
	BINARY_IDENTICAL(0xb6966e59, 0xbdf70a51, "hkpAction"),													// COM-629
	BINARY_IDENTICAL(0xe94d2688, 0x54a4b841, "hkpCdBody"),													// COM-629
	BINARY_IDENTICAL(0x5a47706c, 0xea7f1d08, "hkpShapeInfo"),												// COM-629
	BINARY_IDENTICAL(0x1b58f0ef, 0xff724c17, "hkpPhysicsSystem"),											// COM-629
	{ 0x9d1464fb, 0x6fd7f77f, hkVersionRegistry::VERSION_COPY, "hkpSimpleShapePhantom", HK_NULL },			// COM-629
	{ 0x39409e2d, 0x42aeaf1f, hkVersionRegistry::VERSION_COPY, "hkpWorldObject", HK_NULL },					// COM-629

	{ 0x2743fc9f, 0x8944dc00, hkVersionRegistry::VERSION_COPY, "hkpWorldCinfo", HK_NULL }, // HVK-5026 Added new member.
	{ 0x0d13f6b4, 0xda8c7d7d, hkVersionRegistry::VERSION_COPY, "hkWorldMemoryAvailableWatchDog", HK_NULL },
	{ 0xa7401420, 0x5aad4de6, hkVersionRegistry::VERSION_COPY, "hkpStorageExtendedMeshShapeMeshSubpartStorage", HK_NULL }, // HVK-4699, added new member
	REMOVED("hkpShapeRayBundleCastInput"),																	// COM-629
	REMOVED("hkpShapeRayCastInput"),																		// COM-629
	REMOVED("hkpSimpleShapePhantomCollisionDetail"),														// COM-629
	{ 0x44ff20d2, 0x95fa3c1d, hkVersionRegistry::VERSION_COPY, "hkpConvexVerticesShape", HK_NULL },			// HVK-5079 Added new member.

	// animation
	{ 0x6728e4b7, 0xa8ccd5cf, hkVersionRegistry::VERSION_COPY, "hkaBoneAttachment", Update_hkaBoneAttachment },	// COM-629 700-$$
	{ 0x2a1e146f, 0xb892fd4f, hkVersionRegistry::VERSION_COPY, "hkaSkeleton", Update_hkaSkeleton },			// COM-629 700-$$
	BINARY_IDENTICAL(0xa74011f0, 0x35912f8a, "hkaBone"),													// COM-629 700-$$
	{ 0xf456626d, 0x8dc20333, hkVersionRegistry::VERSION_COPY, "hkaAnimationContainer", HK_NULL },			// COM-629 700-$$
	{ 0xbd7f7a93, 0x66eac971, hkVersionRegistry::VERSION_COPY, "hkaAnimationBinding", HK_NULL },			// COM-629 700-$$
	{ 0x4eae6610, 0x81d9950b, hkVersionRegistry::VERSION_COPY, "hkaMeshBinding", HK_NULL },					// COM-629 700-$$
	{ 0x4da6a6f4, 0x48aceb75, hkVersionRegistry::VERSION_COPY, "hkaMeshBindingMapping", HK_NULL },			// COM-629 700-$$

	{ 0x12ccaefd, 0x95687ea0, hkVersionRegistry::VERSION_COPY, "hkaSkeletonMapperData", HK_NULL },			// HKA-821
	{ 0xeddacc32, 0x12df42a5, hkVersionRegistry::VERSION_COPY, "hkaSkeletonMapper", HK_NULL },				// HKA-821

	{ 0x62b02e7b, 0xf3d472b8, hkVersionRegistry::VERSION_COPY, "hkaInterleavedUncompressedAnimation", Update_hkaInterleavedUncompressedAnimation },	// HKA-1217
	{ 0xf0b3f7d1, 0xff68c1dd, hkVersionRegistry::VERSION_COPY, "hkaDeltaCompressedAnimation", Update_hkaDeltaCompressedAnimation },	// HKA-1217
	{ 0xd1993818, 0x56612989, hkVersionRegistry::VERSION_COPY, "hkaWaveletCompressedAnimation", Update_hkaWaveletCompressedAnimation },	// HKA-1217
	{ 0xba632fd5, 0xd4114fdd, hkVersionRegistry::VERSION_COPY, "hkaAnnotationTrack", Update_hkaAnnotationTrack },	// HKA-1217
	{ 0x731888ca, 0x623bf34f, hkVersionRegistry::VERSION_COPY, "hkaAnnotationTrackAnnotation", HK_NULL },	// HKA-1217
	{ 0x98f9313d, 0x363eb9ef, hkVersionRegistry::VERSION_COPY, "hkaAnimation", Update_hkaAnimation },	// HKA-1217
	{ 0x122f506b, 0x6d85e445, hkVersionRegistry::VERSION_COPY, "hkaDefaultAnimatedReferenceFrame", HK_NULL },	// HKA-1217
	REMOVED( "hkaAnimationPreviewColor" ),																		// HKA-1217  This class is never exported, hence it is not actually versioned
	{ 0x5dad958f, 0x4bc4c3e0, hkVersionRegistry::VERSION_COPY, "hkaAnimationPreviewColorContainer", Update_hkaAnimationPreviewColorContainer },	// HKA-1217  This class is never exported, hence it is not actually versioned
	{ 0x47d3c866, 0x824faf75, hkVersionRegistry::VERSION_COPY, "hkaFootstepAnalysisInfo", HK_NULL },			// HKA-1217  This class is never exported, hence it is not actually versioned
	{ 0x8c131532, 0x1d81207c, hkVersionRegistry::VERSION_COPY, "hkaFootstepAnalysisInfoContainer", HK_NULL }, 	// HKA-1217  This class is never exported, hence it is not actually versioned

	// behavior
	{ 0x9f293169, 0x75dfa1b6, hkVersionRegistry::VERSION_COPY, "hkbBehaviorReferenceGenerator", HK_NULL },			// 700-$$
	{ 0xd7bcf835, 0xb86510d6, hkVersionRegistry::VERSION_COPY, "hkbClipGenerator", Update_hkbClipGenerator },		// 700-$$
	{ 0x6ef61494, 0xe8e20e51, hkVersionRegistry::VERSION_COPY, "hkbCustomTestGenerator", HK_NULL },					// 700-$$
	{ 0xbde631dd, 0x518973a8, hkVersionRegistry::VERSION_COPY, "hkbEvaluateHandleModifier", HK_NULL },
	{ 0xef81f9fc, 0x0ae13223, hkVersionRegistry::VERSION_COPY, "hkbExtractRagdollPoseModifier", HK_NULL },			// 700-$$
	{ 0xa35aa7f4, 0xef747f76, hkVersionRegistry::VERSION_COPY, "hkbFootIkModifier", HK_NULL },						// 700-$$
	{ 0x6be843e0, 0x71ef26c1, hkVersionRegistry::VERSION_COPY, "hkbGeneratorTransitionEffect", HK_NULL },			// 700-$$
	{ 0x2e86fbec, 0xf6e68a6c, hkVersionRegistry::VERSION_COPY, "hkbGetHandleOnBoneModifier", HK_NULL },				// 700-$$
	{ 0x860eed6e, 0x91612e98, hkVersionRegistry::VERSION_COPY, "hkbHandIkModifier", HK_NULL },						// 700-$$
	{ 0xb0e8c27d, 0xd8b6401c, hkVersionRegistry::VERSION_COPY, "hkbHandle", HK_NULL },
	{ 0x3065aa6d, 0xba00c6df, hkVersionRegistry::VERSION_COPY, "hkbKeyframeBonesModifier", HK_NULL },				// 700-$$
	{ 0x575f8ea0, 0x8b625cc0, hkVersionRegistry::VERSION_COPY, "hkbPoweredRagdollControlsModifier", HK_NULL },
	{ 0x689d3e34, 0x46379097, hkVersionRegistry::VERSION_COPY, "hkbRagdollDriverModifier", HK_NULL },				// 700-$$
	{ 0x68dcec5f, 0x750a2d67, hkVersionRegistry::VERSION_COPY, "hkbRigidBodyRagdollControlsModifier", HK_NULL },	// 700-$$
	{ 0x53e8d562, 0xafc3d305, hkVersionRegistry::VERSION_COPY, "hkbSenseHandleModifier", HK_NULL },					// 700-$$
	{ 0x0ab8ae91, 0xa96e7a21, hkVersionRegistry::VERSION_COPY, "hkbSimpleCharacter", HK_NULL },						// 700-$$
	{ 0xd1f819e1, 0x2d995d9c, hkVersionRegistry::VERSION_COPY, "hkbTarget", HK_NULL },
	{ 0x66671cae, 0xe45130db, hkVersionRegistry::VERSION_COPY, "hkbTransitionEffect", HK_NULL },					// 700-$$
	REMOVED( "hkbPoweredRagdollModifier" ),														// 700-$$
	REMOVED( "hkbPoweredRagdollModifierKeyframeInfo" ),											// 700-$$
	REMOVED( "hkbRagdollForceModifier" ),														// 700-$$
	REMOVED( "hkbRigidBodyRagdollModifier" ),													// 700-$$
	REMOVED( "hkbGeneratorOutput" ),
	REMOVED( "hkbGeneratorOutputTrack" ),
	REMOVED( "hkbGeneratorOutputConstTrack" ),
	REMOVED( "hkbGeneratorOutputTrackHeader" ),
	REMOVED( "hkbGeneratorOutputTrackMasterHeader" ),
	REMOVED( "hkbGeneratorOutputTracks" ),
	REMOVED( "hkbStateMachineActiveTransitionInfo" ),
	REMOVED( "hkbStateMachineProspectiveTransitionInfo" ),
	BINARY_IDENTICAL( 0x9c5d4a32, 0xc713064e, "hkbBehaviorGraphStringData" ),					// 700-$$
	BINARY_IDENTICAL( 0x495b1aa,  0x5d2639e5, "hkbCharacterStringData" ),						// 700-$$
	BINARY_IDENTICAL( 0x55261c70, 0xaeeb3b54, "hkbCustomTestGeneratorStruck" ),					// 700-$$
	BINARY_IDENTICAL( 0x29e3fdd2, 0x2bbef407, "hkbDemoConfig" ),								// 700-$$
	BINARY_IDENTICAL( 0x211dd03,  0xa8a2c4d3, "hkbDemoConfigCharacterInfo" ),					// 700-$$
	BINARY_IDENTICAL( 0xe88c506,  0xf687ae66, "hkbDemoConfigTerrainInfo" ),						// 700-$$
	BINARY_IDENTICAL( 0x1ba8f825, 0xa0072df4, "hkbExpressionCondition" ),						// 700-$$
	BINARY_IDENTICAL( 0x3319f023, 0x987fdec,  "hkbExpressionData" ),							// 700-$$
	BINARY_IDENTICAL( 0x2adcd252, 0xde2a264e, "hkbExpressionDataArray" ),						// 700-$$
	BINARY_IDENTICAL( 0x5fc9a58,  0x14dfe1dd, "hkbHandIkModifierHand" ),						// 700-$$
	BINARY_IDENTICAL( 0x7b49ba83, 0x6caa9113, "hkbNamedStringEventPayload" ),					// 700-$$
	BINARY_IDENTICAL( 0x3fccb177, 0x65bdd3a0, "hkbNamedEventPayload" ),							// 700-$$
	BINARY_IDENTICAL( 0xbd5c30ba, 0xb86b19e9, "hkbProjectStringData" ),							// 700-$$
	BINARY_IDENTICAL( 0x4a15eb34, 0x6a5094e3, "hkbSequenceStringData" ),						// 700-$$
	BINARY_IDENTICAL( 0x6c40ed33, 0x7e9f910e, "hkbStateMachineStateInfo" ),						// 700-$$
	BINARY_IDENTICAL( 0xc46650,   0x5ab50487, "hkbStringCondition" ),							// 700-$$
	BINARY_IDENTICAL( 0xb77547bd, 0xed04256a, "hkbStringEventPayload" ),						// 700-$$
	BINARY_IDENTICAL( 0xacb6cbc2, 0x15e04b17, "hkbVariableBindingSet" ),						// 700-$$
	BINARY_IDENTICAL( 0x791fb0b1, 0xa8ea35ae, "hkbVariableBindingSetBinding" ),					// 700-$$
	BINARY_IDENTICAL( 0xbaa06e0b, 0xc9d21eef, "hkVariableTweakingHelper" ),						// 700-$$
	BINARY_IDENTICAL( 0xb927583,  0x6738b5e0, "hkVariableTweakingHelperBoolVariableInfo" ),		// 700-$$
	BINARY_IDENTICAL( 0xb7cb2844, 0xdb61e827, "hkVariableTweakingHelperIntVariableInfo" ),		// 700-$$
	BINARY_IDENTICAL( 0x1408958b, 0x78a255e8, "hkVariableTweakingHelperRealVariableInfo" ),		// 700-$$
	BINARY_IDENTICAL( 0x53765369, 0x3f606072, "hkVariableTweakingHelperVector4VariableInfo" ),	// 700-$$
	
	{ 0xdc0159bf, 0x9d62648b, hkVersionRegistry::VERSION_COPY, "hkbNode", HK_NULL },            // 700-$$


	// Ai class versioning for packfiles is not supported (700-$$)
	{ 0xe9c0b19f, 0x8c06732b, hkVersionRegistry::VERSION_COPY, "hkaiAvoidanceSolverAvoidanceProperties", Update_Ai_Ignore },
	{ 0x8a7d02a2, 0xe6b3f3e1, hkVersionRegistry::VERSION_COPY, "hkaiCharacter", Update_Ai_Ignore },
	{ 0x696fe9b9, 0x43094c8f, hkVersionRegistry::VERSION_COPY, "hkaiNavMeshEdge", Update_Ai_Ignore },
	{ 0x5e62fae7, 0xfea5490a, hkVersionRegistry::VERSION_COPY, "hkaiNavMeshFace", Update_Ai_Ignore },
	{ 0x99fb912f, 0x2436d8de, hkVersionRegistry::VERSION_COPY, "hkaiNavMeshUserEdge", Update_Ai_Ignore },
	{ 0x3440b18a, 0x42bd4278, hkVersionRegistry::VERSION_COPY, "hkaiNavMesh", Update_Ai_Ignore },
	{ 0x835c5c70, 0xf098524a, hkVersionRegistry::VERSION_COPY, "hkaiWorld", Update_Ai_Ignore },
	{ 0x77f485ea, 0xed4a757f, hkVersionRegistry::VERSION_COPY, "hkaiCompoundSilhouette", Update_Ai_Ignore },
	{ 0xb5dea912, 0xde3aae71, hkVersionRegistry::VERSION_COPY, "hkaiPath", Update_Ai_Ignore },
	{ 0x336d2ed9, 0x6c816f91, hkVersionRegistry::VERSION_COPY, "hkaiNavMeshCutter", Update_Ai_Ignore },
	
	// Cloth
	{ 0x9a234ba7, 0xc8919997, hkVersionRegistry::VERSION_COPY, "hclClothState", Update_hclClothState },
	{ 0xc05195fb, 0x4d38a6aa, hkVersionRegistry::VERSION_COPY, "hclClothStateBufferAccess", Update_hclClothStateBufferAccess},
	{ 0xa5838d99, 0xf7fda47f, hkVersionRegistry::VERSION_COPY, "hclClothContainer", HK_NULL },	// COM-629 700-$$
	{ 0x2bcb4769, 0x73cfcd8e, hkVersionRegistry::VERSION_COPY, "hclClothSetupContainer", HK_NULL },	// COM-629 700-$$
	BINARY_IDENTICAL( 0x37d5eb17, 0x8cfcd6e7, "hclStorageSetupMeshVertexChannel" ),				// COM-629 700-$$
	BINARY_IDENTICAL( 0x6dd09190, 0x7362e1a8, "hclStorageSetupMeshTriangleChannel" ),			// COM-629 700-$$
	BINARY_IDENTICAL( 0x56bac312, 0xe3cf5e57, "hclStorageSetupMeshBone" ),						// COM-629 700-$$
	BINARY_IDENTICAL( 0x05f701bf, 0x8f40b6dc, "hclStorageSetupMesh" ),							// COM-629 700-$$
	BINARY_IDENTICAL(0x20ce2359, 0xa5093399, "hclBendLinkSetupObject"),
	BINARY_IDENTICAL(0x41ba6bd3, 0x22c9de49, "hclBlendSetupObject"),
	BINARY_IDENTICAL(0xbbb72c59, 0xc8ba3df5, "hclBonePlanesSetupObject" ),
	BINARY_IDENTICAL(0x1e5b5649, 0x6263c607, "hclBonePlanesSetupObjectGlobalPlane"),
	BINARY_IDENTICAL(0x1903e81f, 0x3f6d2b32, "hclBonePlanesSetupObjectPerParticleAngle"),
	BINARY_IDENTICAL(0x6b7c6d4c, 0x86c3013e, "hclBonePlanesSetupObjectPerParticlePlane"),
	BINARY_IDENTICAL(0x2b832fed, 0x902cfa3c, "hclBufferDefinition"),
	BINARY_IDENTICAL(0x5a13f113, 0x80a9963b, "hclClothData"),
	BINARY_IDENTICAL(0x131c1fcb, 0x6c33e121, "hclClothSetupObject"),
	BINARY_IDENTICAL(0x3a092e04, 0xfd5904f7, "hclClothStateSetupObject"),
	BINARY_IDENTICAL(0xe3c7513e, 0x9ea90df7, "hclCollidable"),
	BINARY_IDENTICAL(0xf3a70ac4, 0xa9d66813, "hclConstraintSet"),
	{ 0xdf4799d0, 0x3293a2e0, hkVersionRegistry::VERSION_COPY, "hclConvexHeightFieldShape", HK_NULL},
	BINARY_IDENTICAL(0x98c04e88, 0xc2b12c5f, "hclDisplayBufferSetupObject"),
	BINARY_IDENTICAL(0x5c5081e0, 0xa5c71dac, "hclHingeSetupObject"),
	BINARY_IDENTICAL(0xa8b2fb29, 0x6d395f6c, "hclLocalRangeSetupObject"),
	BINARY_IDENTICAL(0xaa65ff5d, 0xe8f6c781, "hclMeshMeshDeformSetupObject"),
	BINARY_IDENTICAL(0x49f3d576, 0xed6107c3, "hclMoveFixedParticlesSetupObject"),
	BINARY_IDENTICAL(0x3a5d706a, 0xa340ee1e, "hclNamedSetupMesh"),
	BINARY_IDENTICAL(0xad11312d, 0xd5b4fc3d, "hclNamedTransformSetSetupObject"),
	BINARY_IDENTICAL(0x2dbc6208, 0x77cd00df, "hclOperator"),
	BINARY_IDENTICAL(0x3f79899b, 0xd4bcd9c2, "hclRecalculateNormalsSetupObject"),
	BINARY_IDENTICAL(0xc05345a3, 0x40b16a89, "hclScratchBufferSetupObject"),
	BINARY_IDENTICAL(0xccf608be, 0x86999592, "hclSimClothBufferSetupObject"),
	BINARY_IDENTICAL(0x99c1c86e, 0x2c66aa23, "hclSimClothData"),
	BINARY_IDENTICAL(0x4857132b, 0xf7ca3853, "hclSimClothPose"),
	BINARY_IDENTICAL(0x70c32109, 0xf3ed8bcf, "hclSimClothSetupObject"),
	BINARY_IDENTICAL(0xca231414, 0x1f97826a, "hclSimClothSetupObjectPerInstanceCollidable"),
	BINARY_IDENTICAL(0xef4795c7, 0xdd941865, "hclSimulateSetupObject"),
	BINARY_IDENTICAL(0xd9ea3549, 0x9d691577, "hclSimulationSetupMesh"),
	BINARY_IDENTICAL(0x04cdbafb, 0xa89834af, "hclSimulationSetupMeshMapOptions"),
	BINARY_IDENTICAL(0x63d0678c, 0xe084e995, "hclSkeletonTransformSetSetupObject"),
	BINARY_IDENTICAL(0x2fc848be, 0xa1a131dd, "hclSkinSetupObject"),
	BINARY_IDENTICAL(0x6ca40523, 0x2e845b71, "hclStandardLinkSetupObject"),
	BINARY_IDENTICAL(0x98c04e88, 0xc2b12c5f, "hclStaticDisplayBufferSetupObject"),
	BINARY_IDENTICAL(0x758a3529, 0x37d74018, "hclStretchLinkSetupObject"),
	BINARY_IDENTICAL(0x3f0fe451, 0xd00d5189, "hclToolNamedObjectReference"),
	BINARY_IDENTICAL(0x26e7dfc7, 0x0efbd529, "hclTransformSetDefinition"),
	BINARY_IDENTICAL(0x2ad3d33c, 0x08fc0fc3, "hclTransitionSetupObject"),
	BINARY_IDENTICAL(0x9fa85b6e, 0x8e8b20eb, "hclTriangleSelectionInput"),
	BINARY_IDENTICAL(0x118692a4, 0xf4d555c8, "hclVertexCopySetupObject"),
	BINARY_IDENTICAL(0xe641e54a, 0xf7629ecf, "hclVertexFloatInput"),
	BINARY_IDENTICAL(0xde3b7a9c, 0xa1fa4c55, "hclVertexGatherSetupObject"),
	BINARY_IDENTICAL(0x54b2c07e, 0x4591bbfb, "hclVertexSelectionInput"),
	BINARY_IDENTICAL(0x34bb627f, 0x1b14f727, "hclVolumeConstraintSetupObject"),

	REMOVED("hkaiPathFollowingProperties"),
	REMOVED("hkaiTriangleMesh"),
	REMOVED("hkaiTriangleMeshTriangle"),
	REMOVED("hkaiNavMeshCutterDebugSnapshot"),

	{ 0, 0, 0, HK_NULL, HK_NULL }
};	 

static const hkVersionRegistry::ClassRename s_renames[] =
{
	{ "hkpPhysicsSystemDisplayBinding", "hkpDisplayBindingDataPhysicsSystem" },
	{ "hkpRigidBodyDisplayBinding", "hkpDisplayBindingDataRigidBody" },
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

} // namespace hkCompat_hk660r1_hk700b1

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
