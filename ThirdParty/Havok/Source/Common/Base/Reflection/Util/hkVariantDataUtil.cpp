/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

#include <Common/Base/Reflection/Util/hkVariantDataUtil.h>

#include <Common/Base/Math/Vector/hkIntVector.h>

//#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>
#include <Common/Base/Reflection/hkClass.h>

#include <Common/Base/Reflection/Registry/hkClassNameRegistry.h>
#include <Common/Base/Reflection/Registry/hkTypeInfoRegistry.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>

/* static */int HK_CALL hkVariantDataUtil::calcElementSize(hkClassMember::Type type, const hkClass* cls)
{
	return (type == hkClassMember::TYPE_STRUCT) ? cls->getObjectSize() : hkClassMember::getClassMemberTypeProperties(type).m_size;
}

/* static */hkReal HK_CALL hkVariantDataUtil::getReal(hkClassMember::Type type, const void* data)
{
	switch( type )
	{
		case hkClassMember::TYPE_HALF: return ((const hkHalf*)data)->getReal();
		case hkClassMember::TYPE_REAL: return *(const hkReal*)data;
		default: break;
	}

	HK_ASSERT2(0x56396e26, false, "Unsupported float type.");
	return 0.0f;
}

/* static */void HK_CALL hkVariantDataUtil::setReal(hkClassMember::Type type, void* data, hkReal value)
{
	switch( type )
	{
		case hkClassMember::TYPE_HALF: 
		{
			((hkHalf*)data)->setReal<false>(value); 
			break;
		}
		case hkClassMember::TYPE_REAL: 
		{
			*(hkReal*)data = value;
			break;
		}
		default:
		{
			HK_ASSERT2(0x56396e26, false, "Unsupported float type.");
		}
	}
}

/* static */void HK_CALL hkVariantDataUtil::setHalf(hkClassMember::Type type, void* data, hkHalf value)
{
	switch( type )
	{
		case hkClassMember::TYPE_HALF: 
		{
			*(hkHalf*)data = value; 
			break;
		}
		case hkClassMember::TYPE_REAL: 
		{
			*(hkReal*)data = value;
			break;
		}
		default:
		{
			HK_ASSERT2(0x56396e26, false, "Unsupported float type.");
		}
	}
}

/* static */ hkInt64 hkVariantDataUtil::getInt(hkClassMember::Type type, const void* data)
{
	switch( type )
	{
		case hkClassMember::TYPE_BOOL:
		case hkClassMember::TYPE_CHAR:
		case hkClassMember::TYPE_INT8:
		{
			return *(const hkInt8*)data;
		}
		case hkClassMember::TYPE_UINT8:		return *(const hkUint8*)data;
		case hkClassMember::TYPE_INT16:		return *(const hkInt16*)data; 
		case hkClassMember::TYPE_UINT16:	return *(const hkUint16*)data;
		case hkClassMember::TYPE_INT32:		return *(const hkInt32*)data;
		case hkClassMember::TYPE_UINT32:	return *(const hkUint32*)data;
		case hkClassMember::TYPE_INT64:		return *(const hkInt64*)data;
		case hkClassMember::TYPE_UINT64:	return *(const hkUint64*)data;
		case hkClassMember::TYPE_ULONG:		return (hkInt64)*(hkUlong*)data;
		default:
		{
			HK_ASSERT(0x35d68b15, 0);
			return 0;
		}
	}
}

/* static */ hkInt64 hkVariantDataUtil::getInt(hkClassMember::Type type, hkClassMember::Type subType, const void* data)
{
	switch (type)
	{
		case hkClassMember::TYPE_ENUM:
		case hkClassMember::TYPE_FLAGS:
		{
			return getInt(subType, data);
		}
		default: return getInt(type, data);
	}
}

/* static */ void hkVariantDataUtil::setInt(hkClassMember::Type type, void* data, hkInt64 value)
{
	switch( type )
	{
		case hkClassMember::TYPE_BOOL:
		{
			*(hkInt8*)data = hkInt8(value != 0);
			break;
		}
		case hkClassMember::TYPE_CHAR:
		case hkClassMember::TYPE_INT8:
		{
			*(hkInt8*)data = hkInt8(value);
			break;
		}
		case hkClassMember::TYPE_UINT8:		
		{
			*(hkUint8*)data = hkUint8(value);
			break;
		}
		case hkClassMember::TYPE_INT16:		
		{
			*(hkInt16*)data = hkInt16(value);
			break;
		}
		case hkClassMember::TYPE_UINT16:	
		{
			*(hkUint16*)data = hkUint16(value);
			break;
		}
		case hkClassMember::TYPE_INT32:		
		{
			*(hkInt32*)data = hkInt32(value);
			break;
		}
		case hkClassMember::TYPE_UINT32:	
		{
			*(hkUint32*)data = hkUint32(value);
			break;
		}
		case hkClassMember::TYPE_INT64:		
		{
			*(hkInt64*)data = hkInt64(value);
			break;
		}
		case hkClassMember::TYPE_UINT64:	
		{
			*(hkUint64*)data = hkUint64(value);
			break;
		}
		case hkClassMember::TYPE_ULONG:		
		{
			*(hkUlong*)data = hkUlong(value);
			break;
		}
		default:
		{
			HK_ASSERT(0x35d68b15, 0);
		}
	}
}


/* static */void hkVariantDataUtil::setInt(hkClassMember::Type type, hkClassMember::Type subType, void* data, hkInt64 value)
{
	switch (type)
	{
		case hkClassMember::TYPE_ENUM:
		case hkClassMember::TYPE_FLAGS:
		{
			setInt(subType, data, value);
			break;
		}
		default: 
		{
			setInt(type, data, value);
			break;
		}
	}
}

const char* hkVariantDataUtil::getString(hkClassMember::Type type, const void* data )
{
	switch(type)
	{
		case hkClassMember::TYPE_CSTRING:		return *(const char* const*)data;
		case hkClassMember::TYPE_STRINGPTR:		return ((hkStringPtr*const)data)->cString();
		default:
		{
			HK_ASSERT2(0x4421fb04, false, "Unsupported string type.");
			return HK_NULL;
		}
	}
}

/* static */void hkVariantDataUtil::setString(hkClassMember::Type type, void* data, const char* value )
{
	switch (type)
	{
		case hkClassMember::TYPE_CSTRING:			
		{
			*(const char**)data = value;
			break;
		}
		case hkClassMember::TYPE_STRINGPTR:
		{
			*(hkStringPtr*)data = value;
			break;
		}
		default:
		{
			HK_ASSERT2(0x4421fb04, false, "Unsupported string type.");
		}
	}
}

/* static */hkBool HK_CALL hkVariantDataUtil::needsConstruction(const hkTypeInfoRegistry* infoReg, hkClassMember& mem)
{
	switch (mem.getType())
	{
		case hkClassMember::TYPE_STRUCT:
		{
			const hkTypeInfo* info = infoReg->getTypeInfo(mem.getClass()->getName());

			return info && (info->hasCleanupFunction() || info->hasFinishFunction());
		}
		case hkClassMember::TYPE_ARRAY:
		case hkClassMember::TYPE_VARIANT:
		case hkClassMember::TYPE_POINTER:
		case hkClassMember::TYPE_STRINGPTR:
		{
			return true;
		}
		default:
		{
			return false;
		}
	}
}

/* static */void hkVariantDataUtil::newInstance(const hkClass& cls, void* data)
{	
	// Zero all memory initially
	hkString::memSet(data, 0, cls.getObjectSize());


	if (cls.hasVtable())
	{
		// Set up for ref counting etc.
		hkReferencedObject* refObj = static_cast<hkReferencedObject*>(data);

		HK_ASSERT(0x23432aa2, cls.getObjectSize() > 0);

		refObj->m_memSizeAndFlags = hkUint16(cls.getObjectSize());
		refObj->m_referenceCount = 1;
	}
}

/* static */void hkVariantDataUtil::finishObject(const hkTypeInfoRegistry* infoReg, const hkClass& cls, void* data)
{
	const hkTypeInfo* info = infoReg->getTypeInfo(cls.getName());

	if (info) 
	{
		if (info->hasFinishFunction())
		{
			info->finishLoadedObject(data, true);
		}
	}
}

/* static */void hkVariantDataUtil::finishObjectWithoutTracker(const hkTypeInfoRegistry* infoReg, const hkClass& cls, void* data)
{
	const hkTypeInfo* info = infoReg->getTypeInfo(cls.getName());

	if (info) 
	{
		if (info->hasFinishFunction())
		{
			info->finishLoadedObjectWithoutTracker(data, true);
		}
	}
}

/* static */void hkVariantDataUtil::deleteInstance(const hkTypeInfoRegistry* infoReg, const hkClass& cls, void* data)
{
	const hkTypeInfo* info = infoReg->getTypeInfo(cls.getName());

	if (info && info->hasFinishFunction())
	{
		info->cleanupLoadedObject(data);
	}
}

/* static */ void hkVariantDataUtil::newArray(const hkTypeInfoRegistry* infoReg, const hkClass& cls, void* array, int size, int stride)
{
	const hkTypeInfo* info = infoReg->getTypeInfo(cls.getName());

	if (!info )
	{
		return;
	}

	hkUint8* cur = (hkUint8*)array;
	for (int i = 0; i < size; i++)
	{
		hkString::memSet(cur, 0, cls.getObjectSize());
		if (info->hasFinishFunction())
		{
			info->finishLoadedObjectWithoutTracker((void*)cur, true);
		}

		cur += stride;
	}
}

/* static */ void hkVariantDataUtil::finishObjectArray(const hkTypeInfoRegistry* infoReg, const hkClass& cls, void* array, int size, int stride)
{
	//hkTypeInfoRegistry* infoReg = hkBuiltinTypeRegistry::getInstance().getLoadedObjectRegistry();
	const hkTypeInfo* info = infoReg->getTypeInfo(cls.getName());

	if (!info || !info->hasFinishFunction())
	{
		return;
	}

	hkUint8* cur = (hkUint8*)array;
	for (int i = 0; i < size; i++)
	{
		info->finishLoadedObjectWithoutTracker((void*)cur, true);
		cur += stride;
	}
}


/* static */ void hkVariantDataUtil::newArray(const hkTypeInfoRegistry* infoReg, hkClassMember::Type type, const hkClass* cls, void* array, int size, int stride)
{
	hkUint8* cur = (hkUint8*)array;
	switch (type)
	{
		case hkClassMember::TYPE_STRUCT:
		{
			HK_ASSERT(0x242343ab, cls);
			newArray(infoReg, *cls, array, size, stride);
			break;
		}
		case hkClassMember::TYPE_POINTER:
		{
			for (int i = 0; i < size; i++)
			{
				*(void**)cur = HK_NULL;
				cur += stride;
			}
			break;
		}
		case hkClassMember::TYPE_VARIANT:
		{
			for (int i = 0; i < size; i++)
			{
				hkVariant& var = *(hkVariant*)cur;

				var.m_class = HK_NULL;
				var.m_object = HK_NULL;
				cur += stride;
			}
			break;
		}
		case hkClassMember::TYPE_STRINGPTR:
		{
			for (int i = 0; i < size; i++)
			{
				new (cur) hkStringPtr;
				cur += stride;
			}
			break;
		}
		default: break;
	}
}

/* static */ void hkVariantDataUtil::deleteArray(const hkTypeInfoRegistry* infoReg, const hkClass& cls, void* array, int size, int stride)
{
	const hkTypeInfo* info = infoReg->getTypeInfo(cls.getName());

	if (!info || !info->hasFinishFunction())
	{
		return;
	}

	hkUint8* cur = (hkUint8*)array;
	for (int i = 0; i < size; i++)
	{
		info->cleanupLoadedObject((void*)cur);
		cur += stride;
	}
}

/* static */ void hkVariantDataUtil::deleteArray(const hkTypeInfoRegistry* infoReg, hkClassMember::Type type, const hkClass* cls, void* array, int size, int stride)
{
	hkUint8* cur = (hkUint8*)array;
	switch (type)
	{
		case hkClassMember::TYPE_STRUCT:
		{
			HK_ASSERT(0x32b432b4, cls);
			deleteArray(infoReg, *cls, array, size, stride);
			break;
		}
		case hkClassMember::TYPE_STRINGPTR:
		{
			for (int i = 0; i < size; i++)
			{
				hkStringPtr* stringPtr = (hkStringPtr*)cur;
				stringPtr->~hkStringPtr();
				cur += stride;
			}
			break;
		}
		default: break;
	}
}

/* static */ void hkVariantDataUtil::clearArray(const hkTypeInfoRegistry* infoReg, void* arrayIn, hkClassMember::Type type, const hkClass* cls)
{
	hkArray<hkUint8>& array = *(hkArray<hkUint8>*)arrayIn;

	if (array.getSize() == 0)
	{
		return;
	}

	const int eleSize = calcElementSize(type, cls);

	deleteArray(infoReg, type, cls, array.begin(), array.getSize(), eleSize);

	array.clear();
}

/* static */ void* hkVariantDataUtil::setArraySize(const hkTypeInfoRegistry* infoReg, void* arrayIn, hkClassMember::Type type, const hkClass* cls, int size)
{
	hkArray<hkUint8>& array = *(hkArray<hkUint8>*)arrayIn;
	const int prevSize = array.getSize();
	if (prevSize == size)
	{
		return array.begin();
	}

	const int eleSize = calcElementSize(type, cls);
	HK_ASSERT(0x324234a2, eleSize >= 1);

	if (size < array.getSize())
	{
		deleteArray(infoReg, type, cls, array.begin() + size * eleSize, prevSize - size, eleSize);
	}
	else
	{
		if (array.getCapacity() < size)
		{
			// Make sure there is enough capacity
			hkArrayUtil::_reserve(hkContainerHeapAllocator().get(arrayIn), arrayIn, size, eleSize);
		}

		// Initialize the new members
		newArray(infoReg, type, cls, array.begin() + prevSize * eleSize, size - prevSize, eleSize);
	}

	array.setSizeUnchecked(size);
	return array.begin();
}

/* static */int hkVariantDataUtil::getArraySize(void* arrayIn)
{
	hkArray<hkUint8>& array = *(hkArray<hkUint8>*)arrayIn;
	return array.getSize();
}

/* static */void* hkVariantDataUtil::reserveArray(void* arrayIn, hkClassMember::Type type, const hkClass* cls, int size)
{
	hkArray<hkUint8>& array = *(hkArray<hkUint8>*)arrayIn;
	const int eleSize = calcElementSize(type, cls);
	HK_ASSERT(0x324234a2, eleSize >= 1);

	hkArrayUtil::_reserve(hkContainerHeapAllocator().get(arrayIn), arrayIn, size, eleSize);
	return array.begin();
}

/* static */int hkVariantDataUtil::calcNumReals(hkClassMember::Type type, int tupleSize)
{
	// Set to 1 if not set
	tupleSize = (tupleSize <= 0) ? 1 : tupleSize;

	switch(type)
	{
		case hkClassMember::TYPE_REAL:		
		{
			return 1 * tupleSize;
		}
		case hkClassMember::TYPE_QUATERNION:
		case hkClassMember::TYPE_VECTOR4:	
		{
			return 4 * tupleSize;
		}
		case hkClassMember::TYPE_MATRIX3:
		case hkClassMember::TYPE_ROTATION:
		case hkClassMember::TYPE_QSTRANSFORM:
		{
			HK_COMPILE_TIME_ASSERT(sizeof(hkQsTransform) == 12 * sizeof(hkReal));
			return 12 * tupleSize;
		}
		case hkClassMember::TYPE_MATRIX4:
		case hkClassMember::TYPE_TRANSFORM:
		{
			HK_COMPILE_TIME_ASSERT(sizeof(hkTransform) == 16 * sizeof(hkReal));
			return 16 * tupleSize;
		}
		case hkClassMember::TYPE_HALF:
		{
			return 1 * tupleSize;
		}
		default: return 0;
	}
}

/* static */void hkVariantDataUtil::setReals(hkClassMember::Type type, int tupleSize, const hkReal* src, void* data, int numRealsIn)
{
	const int typeNumReals = calcNumReals(type, tupleSize);
	const int numReals = (numRealsIn < 0) ? typeNumReals : numRealsIn;
	HK_ASSERT(0x234a24a3, numReals <= typeNumReals);

	if (numReals <= typeNumReals)
	{
		if (type == hkClassMember::TYPE_HALF)
		{
			hkHalf* dst = (hkHalf*)data;
			for (int i = 0; i < numReals; i++)
			{
				dst[i].setReal<false>(src[i]);
			}
		}
		else
		{
			hkReal* dst = (hkReal*)data;
			for (int i = 0; i < numReals; i++)
			{
				dst[i] = src[i];
			}
		}
	}
}

/* static */const hkReal* hkVariantDataUtil::getReals(hkClassMember::Type type, int tupleSize, const void* data, hkArray<hkUint8>& buffer)
{
	const int numReals = calcNumReals(type, tupleSize);
	if (numReals <= 0)
	{
		return HK_NULL;
	}
	if (type == hkClassMember::TYPE_HALF)
	{
		const hkHalf* src = (const hkHalf*)data;
		buffer.setSize(sizeof(hkReal) * numReals);
		hkReal* dst = (hkReal*)buffer.begin();
		for (int i = 0; i < numReals; i++)
		{
			dst[i] = src[i].getReal();
		}

		return dst;
	}
	else
	{
		return (const hkReal*)data;
	}
}

/* static */void hkVariantDataUtil::setObject(const hkVariant& var, hkClassMember::Type type, void* data)
{
	switch( type )
	{
		case hkClassMember::TYPE_POINTER:
		{
			// <TODO JS ! Do I need to ref count?
			*(void**)data = var.m_object;
			break;
		}
		case hkClassMember::TYPE_STRUCT:
		{
			// The data should be the same? Ie, it was got with get
			HK_ASSERT(0x2342b34a, var.m_object == data);
			break;
		}
		case hkClassMember::TYPE_VARIANT:
		{
			*(hkVariant*)data = var;
			break;
		}
		default:
		{
			HK_ASSERT(0x4c75f586, 0);
		}
	}
}

/* static */const hkClass* hkVariantDataUtil::findMostDerivedClass(const void* object, const hkVtableClassRegistry* vtable, const hkClassNameRegistry* classReg)
{
	const hkClass* klass = vtable->getClassFromVirtualInstance(object);
	if( klass )
	{
		klass = classReg->getClassByName(klass->getName());
	}
	return klass;
}

/* static */ hkVariant hkVariantDataUtil::getVariantWithMostDerivedClass(const hkClass* klass, const void* object, const hkVtableClassRegistry* vtable, const hkClassNameRegistry* classReg)
{
	HK_ASSERT(0x1b908859, vtable);
	hkVariant v = {HK_NULL, HK_NULL};
	if( klass && object )
	{
		if( klass->hasVtable() )
		{
			const hkClass* mostDerived = findMostDerivedClass(object, vtable, classReg);
			if( mostDerived )
			{
				//HK_ASSERT2(0x89efe456, klass->isSuperClass(*mostDerived) || mostDerived->isSuperClass(*klass),
				//	"The '" << mostDerived->getName() << "' and '" << klass->getName() << "' are unrelated classes for object at " << object << ".");
				v.m_class = mostDerived;
				v.m_object = const_cast<void*>(object);
			}
			else
			{
				HK_WARN_ALWAYS(0x3476d70f, "Could not find the most derived class for virtual object " << klass->getName() << " at " << object << ". The object is replaced with HK_NULL.");
				return v;
			}
		}
		else
		{
			v.m_class = klass;
			v.m_object = const_cast<void*>(object);
		}
	}
	return v;
}

hkVariant hkVariantDataUtil::getObject( hkClassMember::Type type, const hkClass* cls, const hkVtableClassRegistry* vtableReg, const hkClassNameRegistry* classReg, const void* data)
{
	switch( type )
	{
		case hkClassMember::TYPE_POINTER:
		{
			return getVariantWithMostDerivedClass(cls, *(void*const*)data, vtableReg, classReg);
		}
		case hkClassMember::TYPE_STRUCT:
		{
			return getVariantWithMostDerivedClass(cls, data, vtableReg, classReg);
		}
		case hkClassMember::TYPE_VARIANT:
		{

			return *(const hkVariant*)data;
		}
		default:
		{
			HK_ASSERT(0x4c75f586, 0);

			hkVariant v = {0,0};
			return v;
		}
	}
}

/* static */void hkVariantDataUtil::setPointer(const hkClass& cls, void* obj, void** dst, hkBool isReferenced)
{
	if (cls.hasVtable() && isReferenced)
	{
		// Do the addref before the removeref - in case they point to the same thing
		if (obj)
		{
			hkReferencedObject* refObj = static_cast<hkReferencedObject*>(obj);
			refObj->addReference();
		}
		hkReferencedObject* dstObj = (hkReferencedObject*)*dst;
		if (dstObj)
		{
			dstObj->removeReference();
		}
	}

	*dst = obj;
}

/* static */ hk_size_t hkVariantDataUtil::calcBasicElementSize(hkClassMember::Type type, const hkClass* klass)
{
	switch (type)
	{
		default:
		{
			const hkClassMember::TypeProperties& props = hkClassMember::getClassMemberTypeProperties(type);
			return props.m_size;
		}			
		case hkClassMember::TYPE_STRUCT:
		{
			HK_ASSERT(0x3423a432, klass);
			return klass->getObjectSize();
		}
	}
}

/* static */ hk_size_t hkVariantDataUtil::calcElementSize(hkClassMember::Type type, hkClassMember::Type subType, const hkClass* klass, int tupleSize)
{
	hk_size_t basicSize = 0;

	switch (type)
	{
		case hkClassMember::TYPE_ENUM:
		case hkClassMember::TYPE_FLAGS:
		{
			basicSize = calcBasicElementSize(subType, klass);
			break;
		}
		case hkClassMember::TYPE_ARRAY:
		{
			return sizeof(hkArray<hkUint8>);
		}
		default:
		{
			basicSize = calcBasicElementSize(type, klass);
			break;
		}
	}	

	if (tupleSize > 0)
	{
		basicSize *= tupleSize;
	}
	return basicSize;
}

/* static */ void hkVariantDataUtil::convertTypeToInt32Array(hkClassMember::Type srcType, const void* srcIn, hkInt32* dst, int size)
{
	switch (srcType)
	{
		case hkClassMember::TYPE_BOOL:
		{
			convertBoolToTypeArray((const hkBool*)srcIn, hkClassMember::TYPE_INT32, dst, size);
			break;
		}
		case hkClassMember::TYPE_INT32:
		case hkClassMember::TYPE_UINT32:
		{
			hkString::memCpy(dst, srcIn, sizeof(hkInt32) * size);
			break;
		}
		case hkClassMember::TYPE_UINT64:
		case hkClassMember::TYPE_INT64:
		{
			convertInt64ToTypeArray((const hkInt64*)srcIn, hkClassMember::TYPE_INT32, dst, size);
			break;
		}
#define CONVERT(ENUM, TYPE) \
		case hkClassMember::TYPE_##ENUM: \
		{ \
			const TYPE* src = (const TYPE*)srcIn; \
			for (int i = 0; i < size; i++) \
			{ \
				dst[i] = hkInt32(src[i]); \
			} \
			break; \
		} 
		CONVERT(CHAR, hkChar)
		CONVERT(INT8, hkInt8)
		CONVERT(UINT8, hkUint8)
		CONVERT(INT16, hkInt16)
		CONVERT(UINT16, hkUint16)
#undef CONVERT
		default:
		{
			HK_ASSERT2(0x3242343a, false, "Unhandled type conversion");
		}
	}
}

/* static */ void hkVariantDataUtil::convertInt32ToTypeArray(const hkInt32* src, hkClassMember::Type dstType, void* dstIn, int size)
{
	switch (dstType)
	{
		case hkClassMember::TYPE_BOOL:
		{
			convertTypeToBoolArray(hkClassMember::TYPE_INT32, src, (hkBool*)dstIn, size);
			break;
		}
		case hkClassMember::TYPE_INT32:
		case hkClassMember::TYPE_UINT32:
		{
			hkString::memCpy(dstIn, src, sizeof(hkInt32) * size);
			break;
		}
#define CONVERT(ENUM, TYPE) \
		case hkClassMember::TYPE_##ENUM: \
		{ \
			TYPE* dst = (TYPE*)dstIn; \
			for (int i = 0; i < size; i++) \
			{ \
				dst[i] = TYPE(src[i]); \
			} \
			break; \
		} 
		CONVERT(CHAR, hkChar)
		CONVERT(INT8, hkInt8)
		CONVERT(UINT8, hkUint8)
		CONVERT(INT16, hkInt16)
		CONVERT(UINT16, hkUint16)
		CONVERT(INT64, hkInt64)
		CONVERT(UINT64, hkUint64)
	#undef CONVERT
		default:
		{
			HK_ASSERT2(0x3242343a, false, "Unhandled type conversion");
		}
	}
}

/* static */ void hkVariantDataUtil::convertUint32ToTypeArray(const hkUint32* src, hkClassMember::Type dstType, void* dstIn, int size)
{
	switch (dstType)
	{
		case hkClassMember::TYPE_BOOL:
		{
			convertTypeToBoolArray(hkClassMember::TYPE_UINT32, src, (hkBool*)dstIn, size);
			break;
		}
		case hkClassMember::TYPE_INT32:
		case hkClassMember::TYPE_UINT32:
		{
			hkString::memCpy(dstIn, src, sizeof(hkInt32) * size);
			break;
		}
#define CONVERT(ENUM, TYPE) \
		case hkClassMember::TYPE_##ENUM: \
		{ \
			TYPE* dst = (TYPE*)dstIn; \
			for (int i = 0; i < size; i++) \
			{ \
				dst[i] = TYPE(src[i]); \
			} \
			break; \
		} 
		CONVERT(CHAR, hkChar)
		CONVERT(INT8, hkInt8)
		CONVERT(UINT8, hkUint8)
		CONVERT(INT16, hkInt16)
		CONVERT(UINT16, hkUint16)
		CONVERT(INT64, hkInt64)
		CONVERT(UINT64, hkUint64)
	#undef CONVERT
		default:
		{
			HK_ASSERT2(0x3242343a, false, "Unhandled type conversion");
		}
	}
}

/* static */ void hkVariantDataUtil::convertTypeToBoolArray(hkClassMember::Type srcType, const void* srcIn, hkBool* dst, int size)
{
	if (srcType == hkClassMember::TYPE_BOOL)
	{
		hkString::memCpy(dst, srcIn, size * sizeof(hkBool));
		return;
	}

	const hkClassMember::TypeProperties& srcProps = hkClassMember::getClassMemberTypeProperties(srcType);

	switch (srcProps.m_size)
	{
		case 1:
		{
			const hkUint8* src = (const hkUint8*)srcIn;
			for (int i = 0; i < size; i++)
			{
				dst[i] = (src[i] != 0);
			}
			break;
		}
		case 2:
		{
			const hkUint16* src = (const hkUint16*)srcIn;
			for (int i = 0; i < size; i++)
			{
				dst[i] = (src[i] != 0);
			}
			break;
		}
		case 4:
		{
			const hkUint32* src = (const hkUint32*)srcIn;
			for (int i = 0; i < size; i++)
			{
				dst[i] = (src[i] != 0);
			}
			break;
		}
		case 8:
		{
			const hkUint64* src = (const hkUint64*)srcIn;
			for (int i = 0; i < size; i++)
			{
				dst[i] = (src[i] != 0);
			}
			break;
		}
		default:
		{
			HK_ASSERT2(0x3242343a, false, "Unhandled type conversion");
		}
	}
}	

/* static */ void hkVariantDataUtil::convertBoolToTypeArray(const hkBool* src, hkClassMember::Type dstType, void* dstIn, int size)
{
	if (dstType == hkClassMember::TYPE_BOOL)
	{
		hkString::memCpy(dstIn, src, size * sizeof(hkBool));
		return;
	}

	const hkClassMember::TypeProperties& dstProps = hkClassMember::getClassMemberTypeProperties(dstType);
	switch (dstProps.m_size)
	{
		case 1:
		{
			hkUint8* dst = (hkUint8*)dstIn;
			for (int i = 0; i < size; i++)
			{
				dst[i] = (src[i] ? 1 : 0);
			}
			break;
		}
		case 2:
		{
			hkUint16* dst = (hkUint16*)dstIn;
			for (int i = 0; i < size; i++)
			{
				dst[i] = (src[i] ? 1 : 0);
			}
			break;
		}
		case 4:
		{
			hkUint32* dst = (hkUint32*)dstIn;
			for (int i = 0; i < size; i++)
			{
				dst[i] = (src[i] ? 1 : 0);
			}
			break;
		}
		case 8:
		{
			hkUint64* dst = (hkUint64*)dstIn;
			for (int i = 0; i < size; i++)
			{
				dst[i] = (src[i] ? 1 : 0);
			}
			break;
		}
		default:
		{
			HK_ASSERT2(0x3242343a, false, "Unhandled type conversion");
		}
	}
}	

/* static */ void hkVariantDataUtil::convertInt64ToTypeArray(const hkInt64* src, hkClassMember::Type dstType, void* dstIn, int size)
{
	switch (dstType)
	{
		case hkClassMember::TYPE_BOOL:
		{
			convertTypeToBoolArray(hkClassMember::TYPE_INT64, src, (hkBool*)dstIn, size);
			break;
		}
		case hkClassMember::TYPE_INT64:
		case hkClassMember::TYPE_UINT64:
		{
			hkString::memCpy(dstIn, src, sizeof(hkInt64) * size);
			break;
		}
#define CONVERT(ENUM, TYPE) \
		case hkClassMember::TYPE_##ENUM: \
		{ \
			TYPE* dst = (TYPE*)dstIn; \
			for (int i = 0; i < size; i++) \
			{ \
				dst[i] = TYPE(src[i]); \
			} \
			break; \
		} 
		CONVERT(CHAR, hkChar)
		CONVERT(INT8, hkInt8)
		CONVERT(UINT8, hkUint8)
		CONVERT(INT16, hkInt16)
		CONVERT(UINT16, hkUint16)
		CONVERT(INT32, hkInt32)
		CONVERT(UINT32, hkUint32)
	#undef CONVERT
		default:
		{
			HK_ASSERT2(0x3242343a, false, "Unhandled type conversion");
		}
	}
}

/* static */void HK_CALL hkVariantDataUtil::convertTypeToTypeArray(hkClassMember::Type srcType, const void* src, hkClassMember::Type dstType, void* dst, int size)
{
	if (srcType == dstType)
	{
		// Same type.. just copy
		const hkClassMember::TypeProperties& srcProps = hkClassMember::getClassMemberTypeProperties(srcType);
		hkString::memCpy(dst, src, size * srcProps.m_size);
		return;
	}

	switch (srcType)
	{
		case hkClassMember::TYPE_BOOL:
		{
			convertBoolToTypeArray((const hkBool*)src, dstType, dst, size);
			return;
		}
		case hkClassMember::TYPE_UINT64:
		case hkClassMember::TYPE_INT64:
		{
			convertInt64ToTypeArray((const hkInt64*)src, dstType, dst, size);
			return;
		}
		case hkClassMember::TYPE_UINT32:
		{
			convertUint32ToTypeArray((const hkUint32*)src, dstType, dst, size);
			return;
		}
		case hkClassMember::TYPE_INT32:
		{
			convertInt32ToTypeArray((const hkInt32*)src, dstType, dst, size);
			return;
		}
		case hkClassMember::TYPE_INT16:
		case hkClassMember::TYPE_UINT16:
		case hkClassMember::TYPE_INT8:
		case hkClassMember::TYPE_UINT8:
		case hkClassMember::TYPE_CHAR:
		{
			if (dstType == hkClassMember::TYPE_BOOL)
			{
				convertTypeToBoolArray(srcType, src, (hkBool*)dst, size);
				return;
			}

			const hkClassMember::TypeProperties& srcProps = hkClassMember::getClassMemberTypeProperties(srcType);
			const hkClassMember::TypeProperties& dstProps = hkClassMember::getClassMemberTypeProperties(dstType);
			// If integral and same size, then can just copy
			if (srcProps.m_size == dstProps.m_size)
			{
				hkString::memCpy(dst, src, size * srcProps.m_size);
				return;
			}

			// Ugh.. well we can always convert through Int32

			const int maxSize = 64;
			const int srcStride = maxSize * srcProps.m_size;
			const int dstStride = maxSize * dstProps.m_size;

			hkInt32 buffer[maxSize];
			while (size > 0)
			{
				const int curSize = (size <= maxSize) ? size : maxSize;
				convertTypeToInt32Array(srcType, src, buffer, curSize);
				convertInt32ToTypeArray(buffer, dstType, dst, curSize);

				src = hkAddByteOffsetConst(src, srcStride);
				dst = hkAddByteOffset(dst, dstStride);
				size -= curSize;
			}
			return;
		}
		case hkClassMember::TYPE_HALF:
		{
			if (dstType == hkClassMember::TYPE_REAL)
			{
				const hkHalf* s = (const hkHalf*)src;
				hkReal* d = (hkReal*)dst;
				for (int i = 0; i < size; i++)
				{
					d[i] = s[i].getReal();
				}
				return;
			}
			break;
		}
		case hkClassMember::TYPE_REAL:
		{
			if (dstType == hkClassMember::TYPE_HALF)
			{
				const hkReal* s = (const hkReal*)src;
				hkHalf* d = (hkHalf*)dst;
				for (int i = 0; i < size; i++)
				{
					d[i].setReal<false>(s[i]);
				}
				return;
			}
			break;
		}
		default: break;
	}
	HK_ASSERT2(0x3242343a, false, "Unhandled type conversion");
}

/* static */void HK_CALL hkVariantDataUtil::stridedCopy(const void* src, int srcStride, void* dst, int dstStride, int eleSize, int size)
{
	
	if (eleSize == srcStride && eleSize == dstStride)
	{
		// Can just do a copy
		hkString::memCpy(dst, src, eleSize * size);
		return;
	}

	// Okay - it's strided, so we'll have to do something a little different...

	switch (eleSize)
	{
		case 1:
		{
			const hkUint8* s = (const hkUint8*)src;
			hkUint8* d = (hkUint8*)dst;

			for (int i = 0; i < size; i++)
			{
				*d = *s;
				s = hkAddByteOffsetConst(s, srcStride);
				d = hkAddByteOffset(d, dstStride);
			}
			return;
		}
		case 2:
		{
			if ((hk_size_t(src) | hk_size_t(dst) | hk_size_t(srcStride) | hk_size_t(dstStride)) & 1)
			{
				break;
			}
			
			const hkUint16* s = (const hkUint16*)src;
			hkUint16* d = (hkUint16*)dst;

			for (int i = 0; i < size; i++)
			{
				*d = *s;
				s = hkAddByteOffsetConst(s, srcStride);
				d = hkAddByteOffset(d, dstStride);
			}
			return;
		}
		case 4:
		{
			if ((hk_size_t(src) | hk_size_t(dst) | hk_size_t(srcStride) | hk_size_t(dstStride)) & 3)
			{
				break;
			}
			
			const hkUint32* s = (const hkUint32*)src;
			hkUint32* d = (hkUint32*)dst;

			for (int i = 0; i < size; i++)
			{
				*d = *s;
				s = hkAddByteOffsetConst(s, srcStride);
				d = hkAddByteOffset(d, dstStride);
			}
			return;
		}
		case 8:
		{
			if ((hk_size_t(src) | hk_size_t(dst) | hk_size_t(srcStride) | hk_size_t(dstStride)) & 7)
			{
				break;
			}
			
			const hkUint64* s = (const hkUint64*)src;
			hkUint64* d = (hkUint64*)dst;

			for (int i = 0; i < size; i++)
			{
				*d = *s;
				s = hkAddByteOffsetConst(s, srcStride);
				d = hkAddByteOffset(d, dstStride);
			}
			return;
		}
		case 16:
		{
			if ((hk_size_t(src) | hk_size_t(dst) | hk_size_t(srcStride) | hk_size_t(dstStride)) & 15)
			{
				break;
			}
			
			const hkIntVector* s = (const hkIntVector*)src;
			hkIntVector* d = (hkIntVector*)dst;

			for (int i = 0; i < size; i++)
			{
				*d = *s;
				s = hkAddByteOffsetConst(s, srcStride);
				d = hkAddByteOffset(d, dstStride);
			}
			return;
		}
		default: break;
	}
	
	// Do it the simple/slow way
	for (int i = 0; i < size; i++)
	{
		hkString::memCpy(dst, src, eleSize);

		src = hkAddByteOffsetConst(src, srcStride);
		dst = hkAddByteOffset(dst, dstStride);
	}
}

/* static */void HK_CALL hkVariantDataUtil::convertArray(const hkStridedBasicArray& src, const hkStridedBasicArray& dst)
{
	HK_ASSERT(0x2432a432, src.m_size == dst.m_size && src.m_tupleSize == src.m_tupleSize);
	if (src.m_size <= 0)
	{
		return;
	}

	if (src.m_type == dst.m_type && src.m_tupleSize == dst.m_tupleSize)
	{
		const hkClassMember::TypeProperties& srcProps = hkClassMember::getClassMemberTypeProperties(src.m_type);
		stridedCopy(src.m_data, src.m_stride, dst.m_data, dst.m_stride, srcProps.m_size * src.m_tupleSize, src.m_size);
		return;
	}

	const hkClassMember::TypeProperties& srcProps = hkClassMember::getClassMemberTypeProperties(src.m_type);
	const hkClassMember::TypeProperties& dstProps = hkClassMember::getClassMemberTypeProperties(dst.m_type);

	const int srcEleSize = srcProps.m_size * src.m_tupleSize;
	const int dstEleSize = dstProps.m_size * dst.m_tupleSize;

	if (srcEleSize == src.m_stride && dstEleSize == dst.m_stride)
	{
		// Can just do a straight conversion - there are no stride issues
		convertTypeToTypeArray(src.m_type, src.m_data, dst.m_type, dst.m_data, src.m_tupleSize * src.m_size);
		return;
	}

	hkArray<hkUint8>::Temp buffer;

	if (srcEleSize == src.m_stride)
	{
		// The source is aligned
		buffer.setSize(dstEleSize * src.m_size);

		convertTypeToTypeArray(src.m_type, src.m_data, dst.m_type, buffer.begin(), src.m_tupleSize * src.m_size);
		// Do strided copy to destination
		stridedCopy(buffer.begin(), dstEleSize, dst.m_data, dst.m_stride, dstEleSize, src.m_size);
	}
	else
	{
		buffer.setSize(srcEleSize * src.m_size);
		stridedCopy(src.m_data, src.m_stride, buffer.begin(), srcEleSize, srcEleSize, src.m_size);

		// Do conversion
		convertTypeToTypeArray(src.m_type, buffer.begin(), dst.m_type, dst.m_data, src.m_tupleSize * src.m_size);
	}
}

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
