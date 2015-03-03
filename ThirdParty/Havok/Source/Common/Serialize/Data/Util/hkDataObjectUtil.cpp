/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Data/Util/hkDataObjectUtil.h>
#include <Common/Base/Container/RelArray/hkRelArrayUtil.h>
#include <Common/Base/Reflection/Attributes/hkAttributes.h>
#include <Common/Serialize/Data/Util/hkDataObjectToNative.h>
#include <Common/Serialize/Data/Util/hkDataWorldCloner.h>
#include <Common/Serialize/Packfile/hkPackfileData.h>
#include <Common/Serialize/Resource/hkObjectResource.h>
#include <Common/Serialize/Util/hkSerializeMultiMap.h>

#if 0
#	include <Common/Base/Fwd/hkcstdio.h>
#	define LOG(A) printf A; printf("\n");
#else
#	define LOG(A)
#endif

// README for  Copy to native.
// There are two modes of operation. One is to copy into individually refcounted objects.
// The other is to return a packfile data containing all the objects with refcounting disabled.
// We try to reuse the same code for both operations. Thus we do a tiny amount of redundant work
// for each case which may get discarded at the end. Totally worth it for the reuse. One side
// effect is that the packfile data actually contains lots of small allocations instead of a
// flattened memory block; no matter the contract is fulfilled.
//
// The call sequence is something like
// todo = [top]
// while len(todo):
//    get the next item to copy
//    copy it
//    assign pointers which were waiting for this object
//    add pointed objects which are not yet copied to the todo list
// post processing to set the refcounts if using them
// call the finish ctors for all objects
//
// Note that the source can contain references to objects not yet copied.
// When a pointer is encountered, we call resolvePointer to get it immediately
// if the object has been copied already or to schedule the assignment for later
// when the object is copied. The nasty details of the latter are in the TodoList.


//TODO namespace
// namespace {

struct Copier
{
	typedef hkPointerMap<void*, int> ReferenceCountFromObjectPointer;

		// This util class allows us to keep identical code paths for the packfile
		// and the heap copies. The heap version just undoes the work at the end.
	class TrackedData : public hkPackfileData
	{
		public:
			void stopTrackingAllocations()
			{
				HK_ASSERT(0x3b49bb01, m_exports.getSize() == 0);
				HK_ASSERT(0x1dda3dc0, m_imports.getSize() == 0);
				disableDestructors();
				m_memory.clear();
				m_chunks.clear();
			}
	};

	struct LiveObjectInfo
	{
		LiveObjectInfo() : m_refCount(0), m_size(-1) { m_variant.m_object = HK_NULL; m_variant.m_class = HK_NULL; }
		hkVariant m_variant;
		int m_refCount;
		int m_size;
	};

	struct PointerPatch
	{
		void* m_addr;
		hkBool m_isVariant;
		hkBool m_isOwning;
	};

	typedef hkMap<hkDataObject::Handle, int> IndexMapType;
	typedef hkMap<hkDataObject::Handle, LiveObjectInfo> LiveObjectInfoFromHandle;

		// temp progress variables
	hkArray<hkDataObject::Handle> m_todos;
	hkSerializeMultiMap<hkDataObject::Handle, PointerPatch, IndexMapType> m_pointerPatchesFromHandle;
	LiveObjectInfoFromHandle m_liveInfoFromHandle;
	hkArray<hkDataObject::Handle> m_copyStack;
	hkArray<hkVariant> m_postFinishObjects;

		// set at startup
	hkRefPtr<const hkClassNameRegistry> m_classReg;
	hkRefPtr<hkPackfileData> m_trackedData;
	hkBool32 m_allocatedOnHeap;
	hkBool32 m_destroyOnCopy;

	Copier( const hkClassNameRegistry* creg, hkBool32 allocatedOnHeap, hkBool32 destroyOnCopy)
		: m_classReg(creg), m_allocatedOnHeap(allocatedOnHeap), m_destroyOnCopy(destroyOnCopy)
	{
	}

	~Copier()
	{
		HK_ASSERT(0x54052922, m_todos.getSize()==0 );
		HK_ASSERT(0x33c9d0cb, m_pointerPatchesFromHandle.getNumKeys()==0);
		HK_ASSERT(0x67d5f2a4, m_trackedData == HK_NULL);
	}

	HK_FORCE_INLINE hkBool32 isReferencedObject(const hkClass& klass)
	{
		extern const hkClass hkReferencedObjectClass;
		return klass.hasVtable() && hkReferencedObjectClass.isSuperClass(klass);
	}

	void markAsDone( const hkDataObject::Handle& handle, hkVariant& v, int objectSize )
	{
		int refCount = 0;

		// update any pending references to this object
		{
			int firstIndex = m_pointerPatchesFromHandle.getFirstIndex( handle );
			for( int i = firstIndex; i != -1; i = m_pointerPatchesFromHandle.getNextIndex(i) )
			{
				PointerPatch cur = m_pointerPatchesFromHandle.getValue(i);
				if( cur.m_isVariant )
				{
					*static_cast<hkVariant*>(cur.m_addr) = v;
				}
				else
				{
					*static_cast<void**>(cur.m_addr) = v.m_object;
				}
				refCount += cur.m_isOwning;
			}
			m_pointerPatchesFromHandle.removeKey(handle);
		}
		// remember it for future
		{
			LiveObjectInfo info;
			info.m_variant = v;
			info.m_refCount = isReferencedObject( *v.m_class ) ? refCount : 0;
			info.m_size = objectSize;
			m_liveInfoFromHandle.insert( handle, info );
		}
	}

	hkResult resolvePointer( const PointerPatch& patch, const hkDataObject::Handle& handle )
	{
		HK_ASSERT(0x6eca3228, handle.p0 != HK_NULL);
		if( m_allocatedOnHeap && patch.m_isOwning && m_copyStack.indexOf(handle) != -1 )
		{
			HK_WARN(0x19ca1f60, "Circular reference detected, loading as refcounted objects failed.");
			return HK_FAILURE;
		}

		LiveObjectInfoFromHandle::Iterator liveIt = m_liveInfoFromHandle.findKey(handle);
		if( m_liveInfoFromHandle.isValid(liveIt) ) // already copied and alive
		{
			LiveObjectInfo info = m_liveInfoFromHandle.getValue(liveIt);
			if( patch.m_isVariant )
			{
				*static_cast<hkVariant*>(patch.m_addr) = info.m_variant;
			}
			else
			{
				*static_cast<void**>(patch.m_addr) = info.m_variant.m_object;
			}
			info.m_refCount += patch.m_isOwning;
			m_liveInfoFromHandle.setValue(liveIt, info);
		}
		else // haven't seen this object yet, do it later
		{
			if( m_pointerPatchesFromHandle.getFirstIndex(handle) == -1 ) // true if first time seeing the key
			{
				HK_ASSERT(0x590ab53d, handle.p0);
				// Only owning references cause a new todo item. This makes cycle detection
				// much easier. We rely on the fact that an owning reference must exist somewhere.
				if( patch.m_isOwning )
				{
					m_todos.pushBack( handle );
				}
			}
			else if( patch.m_isOwning )
			{
				//If none of the already-seen patches own this handle, we still need to add this to the todo list
				int index = m_pointerPatchesFromHandle.getFirstIndex(handle);

				while(index != -1)
				{
					const PointerPatch& curPatch = m_pointerPatchesFromHandle.getValue(index);
					if( curPatch.m_isOwning )
					{
						break;
					}
					index = m_pointerPatchesFromHandle.getNextIndex(index);
				}

				if( index == -1 ) //We got to the end of the list without seeing an 'owning' patch
				{
					m_todos.pushBack( handle );
				}
			}

			m_pointerPatchesFromHandle.insert(handle, patch);
		}
		return HK_SUCCESS;
	}

	void finishAndTrackObjectsInPackfile( hkPackfileData* trackedData )
	{
		HK_ASSERT(0x7f63fcba, m_todos.getSize()==0);
		for( LiveObjectInfoFromHandle::Iterator it = m_liveInfoFromHandle.getIterator();
			m_liveInfoFromHandle.isValid(it); it = m_liveInfoFromHandle.getNext(it) )
		{
			LiveObjectInfo info = m_liveInfoFromHandle.getValue(it);
			trackedData->trackObject( info.m_variant.m_object, info.m_variant.m_class->getName());
			
			// Keep a list of objects that need a PostFinish step
			const hkClass& klass = *info.m_variant.m_class;
			if( klass.getAttribute( "hk.PostFinish" ) )
			{
				trackedData->m_postFinishObjects.pushBack( info.m_variant );
			}
		}
	}

	void finishAndRefcountLiveObjects(const hkTypeInfoRegistry* typeReg)
	{
		HK_ASSERT(0x7f63fcba, m_todos.getSize()==0);
		for( LiveObjectInfoFromHandle::Iterator it = m_liveInfoFromHandle.getIterator();
			m_liveInfoFromHandle.isValid(it); it = m_liveInfoFromHandle.getNext(it) )
		{
			LiveObjectInfo info = m_liveInfoFromHandle.getValue(it);
			void* ptr = info.m_variant.m_object;
			const hkClass& klass = *info.m_variant.m_class;

			// Keep a list of objects that need a PostFinish step. This saves
			// re-iterating the list
			if(klass.getAttribute("hk.PostFinish"))
			{
				m_postFinishObjects.pushBack(info.m_variant);
			}

			// setRefCountOnObject
			{
				LOG(("FinishAndRefcount '%s'\t\tat 0x%p", klass.getName(), ptr));

				hkInt16 count = (hkInt16)info.m_refCount;
				if( count > 0 )
				{
					HK_ASSERT(0x687dbaae, hkReferencedObjectClass.isSuperClass(klass));
					if(info.m_size > -1)
					{
						HK_ASSERT2(0x7343f387, info.m_size < 0x10000, "Live object too large to convert");
						static_cast<hkReferencedObject*>(ptr)->m_memSizeAndFlags = hkInt16(info.m_size);
					}
					else
					{
						static_cast<hkReferencedObject*>(ptr)->m_memSizeAndFlags = static_cast<hkInt16>(klass.getObjectSize());
					}
					static_cast<hkReferencedObject*>(ptr)->m_referenceCount = count;
				}
			}
			//finishObject
			{
				const hkTypeInfo* typeInfo = typeReg->finishLoadedObject(ptr, klass.getName());
				if( klass.hasVtable() && typeInfo == HK_NULL )
				{
					//return HK_FAILURE;
				}
				//return HK_SUCCESS;
			}
			//HK_WARN_ALWAYS(0x7c303834, "Cannot finish virtual object of class " << obj.m_class->getName() << ". Please make sure the class is registered.");
		}
	}

	void postFinishObjects()
	{
		for( hkArray<hkVariant>::iterator it = m_postFinishObjects.begin(); it < m_postFinishObjects.end(); it++)
		{
			const hkVariant& variant = *it;
			void* ptr = variant.m_object;
			const hkClass& klass = *variant.m_class;

			const hkVariant* attr = klass.getAttribute("hk.PostFinish");
			HK_ASSERT2(0x1e974825, attr && attr->m_class->equals( &hkPostFinishAttributeClass ), "Object does not have PostFinish attribute");
			const hkPostFinishAttribute* postFinishAttr = reinterpret_cast<hkPostFinishAttribute*>(attr->m_object);
			postFinishAttr->m_postFinishFunction(ptr);
		}
	}


	hkResult processPointersAndAllocations(const hkDataObjectToNative::CopyInfoOut& infoOut)
	{
			// remember any allocations
		for( int i = 0; i < infoOut.allocs.getSize(); ++i )
		{
			const hkDataObjectToNative::Alloc& alloc = infoOut.allocs[i];
			if( alloc.m_size != -1 )
			{
				m_trackedData->addChunk(alloc.m_addr, alloc.m_size, HK_MEMORY_CLASS_SERIALIZE);
			}
			else
			{
				m_trackedData->addAllocation(alloc.m_addr);
			}
		}

			// handle any pointers
		for( int i = 0; i < infoOut.pointersOut.getSize(); ++i )
		{
			const hkDataObjectToNative::PointerInfo& info = infoOut.pointersOut[i];
			PointerPatch patch = { info.m_addr, info.m_isVariant, info.m_isOwning };
			if( resolvePointer(patch, info.m_handle) == HK_FAILURE )
			{
				return HK_FAILURE; // circular ref
			}
		}
		return HK_SUCCESS;
	}

	hkVariant abortDeepCopy()
	{
		m_pointerPatchesFromHandle.clear();
		m_todos.clear();
		m_liveInfoFromHandle.clear();
		hkVariant vnull = {HK_NULL,HK_NULL};
		return vnull;
	}

	hkVariant deepCopyToNative( const hkDataObject& topObj ) 
	{
		const hkDataWorld* world = topObj.getClass().getWorld();
		const hkDataObject::Handle hnull = {0,0};

		hkDataObjectToNative copyToNative(m_classReg, m_allocatedOnHeap);
		hkDataObjectToNative::CopyInfoOut infoOut;

		// start the todo list
		hkVariant contents = {0,0};
		{
			PointerPatch patch = { &contents, true, true };
			resolvePointer(patch, topObj.getHandle() );
		}

		while( m_todos.getSize() )
		{
			hkDataObject::Handle curItem = m_todos.back();

			// To detect cycles we need to keep a list of the chain which led to it.
			// We use a trick to know when to pop items off the copyStack - a null todo item.
			if( curItem.p0 == HK_NULL )
			{
				m_todos.popBack();
				m_copyStack.popBack();
				continue;
			}
			m_copyStack.pushBack( curItem );
			m_todos.back() = hnull;

			// Allocate the object body
			hkDataObject dataObj = world->findObject( curItem );
			hkVariant native = copyToNative.allocateObject( dataObj, infoOut );
			HK_ASSERT(0x200c7e8d, (native.m_object==HK_NULL) == (native.m_class==HK_NULL) );
			if( native.m_object == HK_NULL )
			{
				// clean up temp stuff & bug out
				return abortDeepCopy();
			}

			// Move this to the 'done' list and resolve any pending pointers to it.

			int objectSize = -1;
			const hkDataObjectToNative::Alloc& lastAlloc = infoOut.allocs[infoOut.allocs.getSize() - 1];

			if(lastAlloc.m_addr == native.m_object)
			{
				objectSize = lastAlloc.m_size;
			}
			markAsDone( curItem, native, objectSize );

			// Copy members, find pointers
			copyToNative.fillNativeMembers( native.m_object, dataObj, infoOut );

			// Remember allocations made.
			// Resolve pointers or add them to the todo list.
			if( processPointersAndAllocations(infoOut) == HK_SUCCESS )
			{
				infoOut.allocs.clear();
				infoOut.pointersOut.clear();
			}
			else // circular ref, clean up and bug out
			{
				return abortDeepCopy();
			}

			// No longer need the contents of the dataObj
			if (m_destroyOnCopy)
			{
				dataObj.destroy();
			}
		}

		HK_ASSERT2(0x1d4d6659, m_pointerPatchesFromHandle.getNumKeys()==0,
			"There are dangling (unresolved) pointers in the file. This shouldn't happen. "
			"You may ignore this assert and the pointers will be nulled "
			"(and may or may not crash later).");
		return contents;
	}

	hkObjectResource* toObject( const hkDataObject& obj, const hkTypeInfoRegistry* typeReg )
	{
		m_trackedData.setAndDontIncrementRefCount(new hkPackfileData());

		// initial deep copy
		hkVariant ret = deepCopyToNative(obj);

		if( ret.m_object == HK_NULL )
		{
			m_trackedData->disableDestructors();
			m_trackedData = HK_NULL;
			return HK_NULL;
		}
		finishAndRefcountLiveObjects(typeReg);
		postFinishObjects();
		hkObjectResource* res = new hkObjectResource(ret);
		res->setClassNameRegistry(m_classReg);
		res->setTypeInfoRegistry(typeReg);
		static_cast<TrackedData*>(m_trackedData.val())->stopTrackingAllocations();
		m_trackedData = HK_NULL;
		return res;
	}

	HK_FORCE_INLINE hkPackfileData* toPackfile( const hkDataObject& topObj )
	{
		if( topObj.isNull() )
		{
			HK_WARN(0x741f408e, "Can not convert null object to hkPackfileData. Return HK_NULL.");
			return HK_NULL;
		}
		m_trackedData.setAndDontIncrementRefCount(new hkPackfileData(m_classReg));

		// initial deep copy
		hkVariant contents = deepCopyToNative(topObj);
		if( contents.m_object && contents.m_class )
		{
			// succeeded, setup vtables etc
			finishAndTrackObjectsInPackfile(m_trackedData);
			m_trackedData->setContentsWithName( contents.m_object, contents.m_class->getName() );
			hkPackfileData* resource = m_trackedData;
			resource->addReference();
			m_trackedData = HK_NULL;
			return resource;
		}
		m_trackedData = HK_NULL;
		return HK_NULL;
	}
};

// } // namespace

hkObjectResource* HK_CALL hkDataObjectUtil::toObject(const hkDataObject& src, hkBool32 destroyOnCopy)
{
	return toObjectWithRegistry(src, hkBuiltinTypeRegistry::getInstance().getClassNameRegistry(), hkBuiltinTypeRegistry::getInstance().getLoadedObjectRegistry(), destroyOnCopy);
}

hkObjectResource* HK_CALL hkDataObjectUtil::toObjectWithRegistry(const hkDataObject& src, const hkClassNameRegistry* classReg, const hkTypeInfoRegistry* typeReg, hkBool32 destroyOnCopy)
{
	HK_ASSERT(0x693c0602, classReg);
	return Copier(classReg, true, destroyOnCopy).toObject(src, typeReg);
}

hkResource* HK_CALL hkDataObjectUtil::toResource(const hkDataObject& src, hkBool32 destroyOnCopy)
{
	return toResourceWithRegistry(src, hkBuiltinTypeRegistry::getInstance().getClassNameRegistry(), destroyOnCopy);
}

hkResource* HK_CALL hkDataObjectUtil::toResourceWithRegistry(const hkDataObject& src, const hkClassNameRegistry* classReg, hkBool32 destroyOnCopy)
{
	HK_ASSERT(0x693c0602, classReg);
	return Copier(classReg, false, destroyOnCopy).toPackfile(src);
}

namespace // anonymous
{

struct TypeInfo
{
	enum
	{
		TYPE_INVALID,
		TYPE_BASIC,
		TYPE_ARRAY,
		TYPE_ENUM,
		TYPE_TUPLE,
		TYPE_POINTER,
		TYPE_CLASS,
		TYPE_VARIANT,
	};

	hkUint8 m_type;
	hkUint8 m_subType;
	hkUint8 m_tupleSize;
	hkUint8 _pad0;
};

} // anonymous

#define HK_LUT(type, subType, tupleCount) \
{ hkUint8(TypeInfo::TYPE_##type), hkUint8(hkTypeManager::SUB_TYPE_##subType), hkUint8(tupleCount), 0 }

static const TypeInfo s_lut[] =
{
	HK_LUT(BASIC, VOID, 0),			//TYPE_VOID = 0,
	HK_LUT(BASIC, BYTE, 0),			//TYPE_BOOL,
	HK_LUT(BASIC, BYTE, 0),			//TYPE_CHAR,
	HK_LUT(BASIC, INT,	0),			//TYPE_INT8,
	HK_LUT(BASIC, BYTE, 0),			//TYPE_UINT8,
	HK_LUT(BASIC, INT,	0), 		//TYPE_INT16,
	HK_LUT(BASIC, INT,	0),			//TYPE_UINT16,
	HK_LUT(BASIC, INT,	0),			//TYPE_INT32,
	HK_LUT(BASIC, INT,  0),			//TYPE_UINT32,
	HK_LUT(BASIC, INT,  0),			//TYPE_INT64,
	HK_LUT(BASIC, INT,  0),			//TYPE_UINT64,
	HK_LUT(BASIC, REAL, 0),			//TYPE_REAL,

	HK_LUT(TUPLE, REAL, 4),			//TYPE_VECTOR4,
	HK_LUT(TUPLE, REAL, 4),			//TYPE_QUATERNION,
	HK_LUT(TUPLE, REAL,12),			//TYPE_MATRIX3,
	HK_LUT(TUPLE, REAL,12),			//TYPE_ROTATION,
	HK_LUT(TUPLE, REAL,12),			//TYPE_QSTRANSFORM,
	HK_LUT(TUPLE, REAL,16),			//TYPE_MATRIX4,
	HK_LUT(TUPLE, REAL,16),			//TYPE_TRANSFORM,

	HK_LUT(BASIC, VOID, 0),			//TYPE_ZERO,
	HK_LUT(POINTER, INVALID, 0),	//TYPE_POINTER,
	HK_LUT(BASIC, VOID, 0),			//TYPE_FUNCTIONPOINTER,
	HK_LUT(ARRAY, INVALID, 0),		//TYPE_ARRAY,
	HK_LUT(ARRAY, INVALID, 0),		//TYPE_INPLACEARRAY,
	HK_LUT(ENUM,  INVALID, 0),		//TYPE_ENUM,
	HK_LUT(CLASS, INVALID, 0),		//TYPE_STRUCT,
	HK_LUT(ARRAY, INVALID, 0),		//TYPE_SIMPLEARRAY,
	HK_LUT(ARRAY, INVALID, 0),		//TYPE_HOMOGENEOUSARRAY, //TODO: don't hardcode
	HK_LUT(VARIANT, INVALID, 0),		//TYPE_VARIANT,
	HK_LUT(BASIC, CSTRING, 0),		//TYPE_CSTRING,
	HK_LUT(BASIC, INT, 0),			//TYPE_ULONG,
	HK_LUT(ENUM, VOID, 0),			//TYPE_FLAGS,
	HK_LUT(BASIC, REAL, 0),			//TYPE_HALF,
	HK_LUT(BASIC, CSTRING, 0),		//TYPE_STRINGPTR,
	HK_LUT(ARRAY, INVALID, 0), 		//TYPE_RELARRAY
	HK_LUT(BASIC, VOID, 0),			//TYPE_MAX
};
HK_COMPILE_TIME_ASSERT(HK_COUNT_OF(s_lut) == (hkClassMember::TYPE_MAX + 1));

hkDataObject::Type HK_CALL hkDataObjectUtil::getBasicType(hkTypeManager& typeManager, hkClassMember::Type type, const char* typeName)
{
	HK_ASSERT(0x56c8eb98, unsigned(type) < HK_COUNT_OF(s_lut));
	const TypeInfo& info = s_lut[type];

	switch (info.m_type)
	{
		default:
		{
			HK_ASSERT(0x24424332, !"Not a type that is basic");
			return HK_NULL;
		}
		case TypeInfo::TYPE_POINTER:
		{
			if (typeName)
			{
				HK_ASSERT(0x432423a, typeName);
				return typeManager.makePointer(typeManager.addClass(typeName));
			}
			else
			{
				return typeManager.makePointer(typeManager.getHomogenousClass());
			}
		}
		case TypeInfo::TYPE_CLASS:
		{
			HK_ASSERT(0x432423a, typeName);
			return typeManager.addClass(typeName);
		}
		case TypeInfo::TYPE_TUPLE:
		{
			// Its a tuple
			hkDataObject::Type base = typeManager.getSubType(hkTypeManager::SubType(info.m_subType));
			HK_ASSERT(0x243242a3, base);
			return typeManager.makeTuple(base, info.m_tupleSize);
		}
		case TypeInfo::TYPE_BASIC:
		{
			return typeManager.getSubType(hkTypeManager::SubType(info.m_subType));
		}
		case TypeInfo::TYPE_VARIANT:
		{
			return typeManager.makePointer(typeManager.getHomogenousClass());
		}
	}	
}

hkDataObject::Type HK_CALL hkDataObjectUtil::getTypeFromMemberTypeClassName(hkTypeManager& typeManager, hkClassMember::Type mtype, hkClassMember::Type stype, const char* typeName, int count)
{
	const TypeInfo& info = s_lut[mtype];

	hkDataObject::Type type = HK_NULL;

	switch (info.m_type)
	{
		case TypeInfo::TYPE_ENUM:
		{
			type = getBasicType(typeManager, stype, HK_NULL);
			break;
		}
		case TypeInfo::TYPE_CLASS:
		{
			// Struct
			type = getBasicType(typeManager, mtype, typeName);
			break;
		}
		case TypeInfo::TYPE_POINTER:
		{
			type = typeManager.makePointer(getBasicType(typeManager, stype, typeName));
			break;
		}
		case TypeInfo::TYPE_ARRAY:
		{
			// Its an array
			if (stype == hkClassMember::TYPE_STRUCT)
			{
				HK_ASSERT(0x23423432, typeName);
				type = typeManager.addClass(typeName);
			}
			else
			{
				type = getBasicType(typeManager, stype, typeName);
			}

			// Tuple takes place before 'arrayizing'
			if( count != 0 )
			{
				type = typeManager.makeTuple(type, count);
			}

			// Its an array 
			return typeManager.makeArray(type);
		}
		case TypeInfo::TYPE_BASIC:
		{
			// Should be a basic type
			type = typeManager.getSubType(hkTypeManager::SubType(info.m_subType));
			break;
		}
		case TypeInfo::TYPE_TUPLE:
		{
			type = typeManager.makeTuple(typeManager.getSubType(hkTypeManager::SubType(info.m_subType)), info.m_tupleSize);
			break;
		}
		case TypeInfo::TYPE_VARIANT:
		{
			type = getBasicType(typeManager, mtype, typeName);
			break;
		}
	}

	HK_ASSERT(0x2342423, type);
	if( count != 0 )
	{
		type = typeManager.makeTuple(type, count);
	}
	return type;
}

hkDataObject::Type HK_CALL hkDataObjectUtil::getTypeFromMemberType(hkTypeManager& typeManager, hkClassMember::Type mtype, hkClassMember::Type stype, const hkClass* klass, int count)
{
	if (klass)
	{
		return getTypeFromMemberTypeClassName(typeManager, mtype, stype, klass->getName(), count);
	}
	else
	{
		return getTypeFromMemberTypeClassName(typeManager, mtype, stype, HK_NULL, count);
	}
}

hkResult HK_CALL hkDataObjectUtil::deepCopyWorld( hkDataWorld& dst, const hkDataWorld& src)
{
	return hkDataWorldCloner(dst, src).clone();
}

int HK_CALL hkDataObjectUtil::getExtraStorageSize(const hkClass* klass, const hkDataObject& obj)
{
	HK_ASSERT(0x4259cbbf, klass);

	int totalReturn = 0;
	for(int i=0;i<klass->getNumMembers();i++)
	{
		const hkClassMember& member = klass->getMember(i);
		if(member.getType() == hkClassMember::TYPE_RELARRAY)
		{
			hkDataArray objArray(obj[member.getName()].asArray());
			totalReturn += HK_NEXT_MULTIPLE_OF(hkRelArrayUtil::RELARRAY_ALIGNMENT, objArray.getSize() * member.getArrayMemberSize());
		}
	}
	return totalReturn;
}


#include <Common/Base/Container/PointerMap/hkMap.cxx>
template class hkMapBase<hkDataObject::Handle, hkDataObject::Handle>;
template class hkMap<hkDataObject::Handle, hkDataObject::Handle>;

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
