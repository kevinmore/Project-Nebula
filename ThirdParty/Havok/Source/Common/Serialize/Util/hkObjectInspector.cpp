/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Util/hkObjectInspector.h>
#include <Common/Base/Container/RelArray/hkRelArray.h>

#if 0
extern "C" int printf(const char*,...);
#	define PRINT(A) printf A
#else
#	define PRINT(A) /* nothing */
#endif

using hkObjectInspector::Pointer;
typedef hkArray<hkObjectInspector::Pointer>::Temp PointerArray;

#if 0
// NOTE: This may be used potentially in the hkObjectInspector::getObjectsList implementation.
// The idea is to use BuiltinTypeRegistry to resolve hkClass* for virtual objects.
static inline const hkClass* lookupKlassPointer(const void* objPtr, const hkClass* objKlass, const hkVtableClassRegistry& registry)
{
	HK_ASSERT(0x15a09058, objKlass != HK_NULL);
	if (objPtr && objKlass->hasVtable())
	{
		const hkClass* vKlass = registry.getClassFromVirtualInstance(objPtr);
		return (vKlass != HK_NULL ? vKlass : objKlass);
	}
	return objKlass;
}

namespace {
	// Collects information about objects into hkVariant array.
	class CollectObjectsListener: public ObjectListener
	{
	public:
		CollectObjectsListener(VariantArray& collectedObjects)
		: m_collectedObjects(collectedObjects) { }
		// This listener collects and holds a list objects.
		virtual hkResult objectCallback( const void* objP, const hkClass& klass, PointerArray& containedPointers );

	private:
		hkArray<hkVariant>& m_collectedObjects;
		hkPointerMap<const void*, const hkClass*> m_trackedObjects;
	};
}

hkResult CollectObjectsListener::objectCallback( const void* objP, const hkClass& klass, PointerArray& containedPointers )
{
	if (m_trackedObjects.hasKey(objP) == false)
	{
		hkVariant& v = m_collectedObjects.expandOne();
		m_trackedObjects.insert(objP, &klass);
		v.m_class = &klass;
		v.m_object = const_cast<void*>(objP);
	}
	return HK_SUCCESS;
}
#endif // 0

static inline void scanArrayOfPointers(const void* arrayStart, int arraySize, const hkClass* klass, PointerArray& ioPointers);
static inline void scanArrayOfVariants(const void* arrayStart, int arraySize, PointerArray& ioPointers);
static hkResult scanArrayOfStructs(const void* arrayStart, int arraySize, const hkClass* klass, PointerArray& ioPointers);
static inline hkResult scanBody(const void* data, const hkClass& klass, PointerArray& ioPointers);

extern const hkClass hkClassClass;
namespace
{
	template <typename T>
	struct ObjectInspector_DummyArray
	{
		T* data;
		int size;
		int capAndFlags;
	};

	struct ObjectInspector_DummyHomogeneousArray
	{
		hkClass* klass;
		void* data;
		int size;
		//		int capAndFlags;
	};
}

static inline int calcCArraySize( const hkClassMember& member )
{
	return (member.getCstyleArraySize()) ? member.getCstyleArraySize() : 1;
}

inline void scanArrayOfPointers(const void* arrayStart, int arraySize, const hkClass* klass, PointerArray& ioPointers)
{
	HK_COMPILE_TIME_ASSERT( sizeof(void*) <= 8 );

	int itemSize = sizeof(void*);
	Pointer* p = ioPointers.expandBy(arraySize);
	for (int i = 0; i < arraySize; ++i)
	{
		void* vp = const_cast<char*>(static_cast<const char*>(arrayStart)+i*itemSize);
		void** locPtr = static_cast<void**>(vp);
		p[i].location = locPtr;
		p[i].klass = klass; // potentially: const_cast<hkClass*>(lookupKlassPointer(*locPtr, klass));
	}
}

inline void scanArrayOfVariants(const void* arrayStart, int arraySize, PointerArray& ioPointers)
{
	Pointer* p = ioPointers.expandBy(arraySize);
	const hkVariant* v = static_cast<const hkVariant*>(arrayStart);
	for( int i = 0; i < arraySize; ++i )
	{
		void* variantLocation = const_cast<void*>(static_cast<const void*>(v + i));
		p[i].location = static_cast<void**>(variantLocation);
		p[i].klass = v[i].m_class;
	}
}

static hkResult scanArrayOfStructs(const void* arrayStart, int arraySize, const hkClass* klass, PointerArray& ioPointers)
{
	int itemSize = klass->getObjectSize();
	for (int i = 0; i < arraySize; ++i)
	{
		const void* objPtr = static_cast<const char*>(arrayStart)+i*itemSize;
		if (hkObjectInspector::getPointers(objPtr, *klass, ioPointers) == HK_FAILURE)
		{
			return HK_FAILURE;
		}
	}
	return HK_SUCCESS;
}

inline hkResult scanBody(const void* data, const hkClass& klass, PointerArray& containedPointers)
{
	for( int memberIdx = 0; memberIdx < klass.getNumMembers(); ++memberIdx )
	{
		const hkClassMember& member = klass.getMember( memberIdx );

		const void* memberStart = static_cast<const char*>(data) + member.getOffset();

		switch( member.getType() )
		{
			case hkClassMember::TYPE_BOOL:
			case hkClassMember::TYPE_CHAR:
			case hkClassMember::TYPE_INT8:
			case hkClassMember::TYPE_UINT8:
			case hkClassMember::TYPE_INT16:
			case hkClassMember::TYPE_UINT16:
			case hkClassMember::TYPE_INT32:
			case hkClassMember::TYPE_UINT32:
			case hkClassMember::TYPE_INT64:
			case hkClassMember::TYPE_UINT64:
			case hkClassMember::TYPE_ULONG:
			case hkClassMember::TYPE_REAL:
			case hkClassMember::TYPE_HALF:
			case hkClassMember::TYPE_VECTOR4:
			case hkClassMember::TYPE_QUATERNION:
			case hkClassMember::TYPE_MATRIX3:
			case hkClassMember::TYPE_ROTATION:
			case hkClassMember::TYPE_QSTRANSFORM:
			case hkClassMember::TYPE_MATRIX4:
			case hkClassMember::TYPE_TRANSFORM:
			case hkClassMember::TYPE_ENUM:
			case hkClassMember::TYPE_ZERO:
			case hkClassMember::TYPE_CSTRING:
			case hkClassMember::TYPE_STRINGPTR:
			case hkClassMember::TYPE_FUNCTIONPOINTER:
			case hkClassMember::TYPE_FLAGS:
			{
				break;
			}
			case hkClassMember::TYPE_POINTER:
			{
				HK_COMPILE_TIME_ASSERT( sizeof(void*) <= 8 );

				if (member.getSubType() == hkClassMember::TYPE_STRUCT)
				{
					const void* objPtr = static_cast<const char*>(memberStart);
					objPtr = *static_cast<const void*const*>(objPtr);
					scanArrayOfPointers(memberStart, calcCArraySize( member ), &member.getStructClass(), containedPointers);
				}
				break;
			}
			case hkClassMember::TYPE_ARRAY:
			case hkClassMember::TYPE_SIMPLEARRAY:
			case hkClassMember::TYPE_INPLACEARRAY:
			{
				HK_COMPILE_TIME_ASSERT( sizeof(void*) <= 8 );

				const ObjectInspector_DummyArray<char>* carray = (const ObjectInspector_DummyArray<char>*)memberStart;
				hkClassMember::Type atype = member.getArrayType();
				if (carray->data != HK_NULL)
				{
					if (atype == hkClassMember::TYPE_POINTER)
					{
						scanArrayOfPointers(carray->data, carray->size, &member.getStructClass(), containedPointers);
					}
					else if (atype == hkClassMember::TYPE_STRUCT)
					{
						if (scanArrayOfStructs(carray->data, carray->size, &member.getStructClass(), containedPointers) == HK_FAILURE)
						{
							return HK_FAILURE;
						}
					}
					else if (atype == hkClassMember::TYPE_VARIANT)
					{
						const ObjectInspector_DummyArray<hkVariant>* varray = (const ObjectInspector_DummyArray<hkVariant>*)memberStart;
						scanArrayOfVariants(varray->data, varray->size, containedPointers);
					}
				}
				break;
			}

			case hkClassMember::TYPE_RELARRAY:
			{
				const hkRelArray<hkUint8>* ar = (const hkRelArray<hkUint8>*) memberStart;

				hkClassMember::Type atype = member.getArrayType();
				if (ar->getSize() != 0)
				{
					if (atype == hkClassMember::TYPE_POINTER)
					{
						scanArrayOfPointers(ar->begin(), ar->getSize(), &member.getStructClass(), containedPointers);
					}
					else if (atype == hkClassMember::TYPE_STRUCT)
					{
						if (scanArrayOfStructs(ar->begin(), ar->getSize(), &member.getStructClass(), containedPointers) == HK_FAILURE)
						{
							return HK_FAILURE;
						}
					}
					else if (atype == hkClassMember::TYPE_VARIANT)
					{
						const hkRelArray<hkVariant>* varray = (const hkRelArray<hkVariant>*)memberStart;
						scanArrayOfVariants(varray->begin(), varray->getSize(), containedPointers);
					}
				}
				break;
			}

			case hkClassMember::TYPE_HOMOGENEOUSARRAY:
			{
				// class ptr, data ptr, size
				const ObjectInspector_DummyHomogeneousArray* darray = (const ObjectInspector_DummyHomogeneousArray*)memberStart;
			
				if (darray->data != HK_NULL)
				{
					// the class
					const hkClass* structKlass = darray->klass;
					// we can walk through objects with known description only
					if (structKlass != HK_NULL)
					{
						// the data
						if (scanArrayOfStructs(darray->data, darray->size, structKlass, containedPointers) == HK_FAILURE)
						{
							return HK_FAILURE;
						}
					}
				}
				break;
			}
			case hkClassMember::TYPE_STRUCT: // single struct
			{
				const hkClass* structKlass = &member.getStructClass();
				if (scanArrayOfStructs(memberStart, calcCArraySize(member), structKlass, containedPointers) == HK_FAILURE)
				{
					return HK_FAILURE;
				}
				break;
			}
			case hkClassMember::TYPE_VARIANT:
			{
				scanArrayOfVariants(memberStart, calcCArraySize(member), containedPointers);
				break;
			}
			default:
			{
				HK_ERROR(0x641e3e03, "Unknown class member found during write of data.");
				return HK_FAILURE;
			}
		}
	}
	return HK_SUCCESS;
}

hkResult hkObjectInspector::getPointers(const void* object, const hkClass& klass, PointerArray& outList)
{
	return scanBody(object, klass, outList);
}
#if 0
hkResult hkObjectInspector::getObjectsList(const void* object, const hkClass& klass, const hkVtableClassRegistry& registry, VariantArray& outList)
{
	CollectObjectsListener	listener(outList);

	return walkPointers(object, klass, registry, &listener);
}
#endif // 0
hkResult hkObjectInspector::walkPointers(const void* object, const hkClass& klass, ObjectListener* listener)
{
	PointerArray listOfPointers; listOfPointers.reserve(128);
	if( getPointers(object, klass, listOfPointers) == HK_SUCCESS
		&& listener->objectCallback(object, klass, listOfPointers) == HK_SUCCESS )
	{
		for (int i = 0; i < listOfPointers.getSize(); ++i)
		{
			HK_ASSERT2(0x15a09058, *(listOfPointers[i].location) == HK_NULL || ( *(listOfPointers[i].location) != HK_NULL && listOfPointers[i].klass != HK_NULL ),
						"Cannot walk through an object of unknown type." );
			if( *(listOfPointers[i].location) != HK_NULL
				&& walkPointers(*(listOfPointers[i].location), *listOfPointers[i].klass, listener) == HK_FAILURE )
			{
				return HK_FAILURE;
			}
		}	
		return HK_SUCCESS;
	}
	return HK_FAILURE;
}

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
