/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Data/Util/hkDataObjectToNative.h>
#include <Common/Base/Container/RelArray/hkRelArrayUtil.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Reflection/Attributes/hkAttributes.h>
#include <Common/Base/Reflection/Util/hkVariantDataUtil.h>
#include <Common/Serialize/Data/Util/hkDataObjectUtil.h>
#include <Common/Serialize/Util/hkChainedClassNameRegistry.h>

#include <Common/Base/Monitor/hkMonitorStream.h>


#if 0
#	include <Common/Base/Fwd/hkcstdio.h>
#	define LOG(A) printf A; printf("\n");
#else
#	define LOG(A)
#endif

#if defined(HK_DEBUG)
	static inline char* getMemberTypeName(const hkClassMember& member, char* typeName, int size)
	{
		member.getTypeName(typeName, size);
		hkStringBuf tmp(typeName);
		tmp.replace("&lt;", "<");
		tmp.replace("&gt;",">");
		hkString::strCpy(typeName, tmp.cString());
		return typeName;
	}
	#include <Common/Base/Memory/System/Debug/hkDebugMemorySystem.h>
#	define HK_DEBUG_TAG_ADDRESS(PTR, TAG) { if (hkDebugMemorySystem * dms = hkMemorySystem::getInstance().getDebugInterface()) {dms->tagAddress(PTR, TAG);} }
#else
	#define HK_DEBUG_TAG_ADDRESS(PTR, TAG)
#endif // HK_DEBUG

static inline int getNumElementsInMember( const hkClassMember& member )
{
	int asize = member.getCstyleArraySize();
	return asize ? asize : 1;
}

const hkClass* hkDataObjectToNative::findClassOf( const hkDataObject& obj )
{
	const char* className = obj.getClass().getName();
	HK_ASSERT(0x36702c2f, className);
	const hkClass* klass = m_classReg->getClassByName(className);
	HK_ASSERT(0x384740d5, klass != HK_NULL );
	return klass;
}

hkVariant hkDataObjectToNative::allocateObject( const hkDataObject& obj, CopyInfoOut& infoOut )
{
	hkVariant vnull = { HK_NULL, HK_NULL };
	if( obj.getClass().isNull() || obj.getClass().getName() == HK_NULL )
	{
		HK_WARN(0x7b2ae34d, "Found variant referencing object of removed class. Ignore it.");
		return vnull;
	}

	if( const hkClass* klass = m_classReg->getClassByName( obj.getClass().getName() ) )
	{
		if( klass->getDescribedVersion() != obj.getClass().getVersion() )
		{
			HK_WARN(0x41dc1617, "Missing patches. Trying to copy " << klass->getName() << ':' << obj.getClass().getVersion() << " into " << klass->getDescribedVersion() );
		}
		int objectSize = klass->getObjectSize();
		objectSize += hkDataObjectUtil::getExtraStorageSize(klass, obj);
		void* object = hkMemHeapBlockAlloc<char>(objectSize);
		infoOut.allocs.expandOne().set(object, objectSize);

		HK_DEBUG_TAG_ADDRESS(object, klass);
		hkString::memSet(object, 0, objectSize);

		LOG(("0x%p Allocate Object '%s' %i bytes", object, klass->getName(), objectSize));
		hkVariant ret = { object, klass };
		return ret;
	}
	HK_ASSERT3(0x282fb828, 0, "Class " << obj.getClass().getName() << " is not registered. If this is a Havok class, make sure the class's product reflection is enabled near where hkProductFeatures.cxx is included. Otherwise, check your own class registration." );
	
	return vnull;
}

static hkBool32 canFastCopy(hkClassMember::Type type)
{
	switch (type) 
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
		case hkClassMember::TYPE_REAL:
		case hkClassMember::TYPE_VECTOR4:
		case hkClassMember::TYPE_QUATERNION:
		case hkClassMember::TYPE_MATRIX3:

		case hkClassMember::TYPE_ROTATION:
		case hkClassMember::TYPE_QSTRANSFORM:
		case hkClassMember::TYPE_MATRIX4:
		case hkClassMember::TYPE_TRANSFORM:

		//case hkClassMember::TYPE_HALF:

		case hkClassMember::TYPE_ULONG:
		{
			return true;
		}
		default: break;
	}

	return false;
}

hkResult hkDataObjectToNative::copyIntoNativeArray
	( void* address
	, const hkClassMember& member
	, const hkDataArray& srcArray
	, CopyInfoOut& copyInfoOut )
{
	HK_ASSERT( 0x302c7fc9, !m_allocatedOnHeap || member.getType() != hkClassMember::TYPE_SIMPLEARRAY ); // packfile data support only
	HK_ASSERT(0x2343243a, member.getCstyleArraySize() <= 1);

	DummyArray dstArray = { HK_NULL, -1, -1 };
	if( srcArray.getSize() == 0 )
	{
		if( member.getType() == hkClassMember::TYPE_ARRAY ) // init the array capacity
		{
			dstArray.capacity = hkArray<char>::DONT_DEALLOCATE_FLAG;
		}
		return HK_SUCCESS;
	}

	const hkClassMember::Type dstType = member.getSubType();
	const hkClassMember::TypeProperties& dstProps = hkClassMember::getClassMemberTypeProperties(dstType);

	{
		hkStridedBasicArray srcStrided;
		if (dstProps.m_size > 0 && canFastCopy(dstType) && srcArray.asStridedBasicArray(srcStrided) == HK_SUCCESS)
		{

			const int dstElemSize = dstProps.m_size;

			dstArray.size = srcStrided.m_size;

			// Allocate
			int n = srcStrided.m_size * dstElemSize;
			dstArray.data = hkMemoryRouter::getInstance().heap().bufAlloc(n);
			dstArray.capacity = n / dstElemSize;

			// Set up the dst strided structure
			hkStridedBasicArray dstStrided;
			dstStrided.m_type = dstType;
			dstStrided.m_size = srcStrided.m_size;
			dstStrided.m_stride = dstElemSize;
			dstStrided.m_tupleSize = 1;
			dstStrided.m_data = dstArray.data;

			switch (dstType)
			{
				case hkClassMember::TYPE_QUATERNION:
				case hkClassMember::TYPE_VECTOR4:
				{
					dstStrided.m_type = hkClassMember::TYPE_REAL;
					dstStrided.m_tupleSize = 4;
					break;
				}
				case hkClassMember::TYPE_QSTRANSFORM:
				case hkClassMember::TYPE_ROTATION:
				case hkClassMember::TYPE_MATRIX3:
				{
					dstStrided.m_type = hkClassMember::TYPE_REAL;
					dstStrided.m_tupleSize = 12;
					break;
				}
				case hkClassMember::TYPE_TRANSFORM:
				case hkClassMember::TYPE_MATRIX4:
				{
					dstStrided.m_type = hkClassMember::TYPE_REAL;
					dstStrided.m_tupleSize = 16;
					break;
				}
				default: break;
			}

			hkVariantDataUtil::convertArray(srcStrided, dstStrided);

			HK_ASSERT2(0x667acc1d, (dstElemSize > 0) == (dstArray.data!=HK_NULL), "internal consistency error" );

			if( dstElemSize > 0 )
			{
				DummyArray& dst = *static_cast<DummyArray*>(address);
				dst.data = dstArray.data;
				dst.size = dstArray.size;
				copyInfoOut.addAlloc( dstArray.data, dstElemSize * dstArray.size );
				LOG(("0x%p Allocate hkArray/hkSimpleArray %i elems, %i bytes", dstArray.data, dstArray.size, dstElemSize * dstArray.size));
			}
			// Not required for hkRelArray as there is no extra allocation
			if( member.getType() == hkClassMember::TYPE_ARRAY )
			{
				if( !m_allocatedOnHeap )
				{
					dstArray.capacity |= hkArray<char>::DONT_DEALLOCATE_FLAG;
				}
				DummyArray& dst = *static_cast<DummyArray*>(address);
				dst.capacity = dstArray.capacity;
			}
			return HK_SUCCESS;
		}
	}

	int sizeofElem = 0;

	hkResult result = HK_SUCCESS;
	hkBool32 owningReference = member.getFlags().get(hkClassMember::NOT_OWNED) == 0;

	const int size = srcArray.getSize();

	switch( member.getSubType() )
	{
		case hkClassMember::TYPE_BOOL:
		{
			dstArray.size = size;
			dstArray.capacity = dstArray.size;
			dstArray.data = hkMemHeapBufAlloc<hkBool>(dstArray.capacity);
			sizeofElem = sizeof(hkBool);
			
			for( int i = 0; i < size; ++i )
			{
				int val = srcArray[i].asInt();
				static_cast<hkBool*>(dstArray.data)[i] = (val != 0);
			}
			break;
		}
#define CASE_INTEGER(ENUM, TYPE) \
		case hkClassMember::ENUM: \
		{ \
			dstArray.size = size; \
			dstArray.capacity = dstArray.size; \
			dstArray.data = hkMemHeapBufAlloc<TYPE>(dstArray.capacity); \
			sizeofElem = sizeof(TYPE); \
			for( int i = 0; i < size; ++i ) \
			{ \
				static_cast<TYPE*>(dstArray.data)[i] = static_cast<TYPE>(srcArray[i].asInt()); \
			} \
		} \
		break

#define CASE_64BIT_INTEGER(ENUM, TYPE) \
		case hkClassMember::ENUM: \
		{ \
			dstArray.size = size; \
			dstArray.capacity = dstArray.size; \
			dstArray.data = hkMemHeapBufAlloc<TYPE>(dstArray.capacity); \
			sizeofElem = sizeof(TYPE); \
			for( int i = 0; i < size; ++i ) \
			{ \
				static_cast<TYPE*>(dstArray.data)[i] = static_cast<TYPE>(srcArray[i].asInt64()); \
			} \
		} \
		break


		CASE_INTEGER(TYPE_CHAR, char);
		CASE_INTEGER(TYPE_INT8, hkInt8);
		CASE_INTEGER(TYPE_UINT8, hkUint8);
		CASE_INTEGER(TYPE_INT16, hkInt16);
		CASE_INTEGER(TYPE_UINT16, hkUint16);
		CASE_INTEGER(TYPE_INT32, hkInt32);
		CASE_INTEGER(TYPE_UINT32, hkUint32);

		CASE_64BIT_INTEGER(TYPE_INT64, hkInt64);
		CASE_64BIT_INTEGER(TYPE_UINT64, hkUint64);
		CASE_64BIT_INTEGER(TYPE_ULONG, hkUlong);
#undef CASE_INTEGER
#undef CASE_64BIT_INTEGER

		case hkClassMember::TYPE_REAL:
		{
			dstArray.size = size;
			dstArray.capacity = dstArray.size;
			dstArray.data = hkMemHeapBufAlloc<hkReal>(dstArray.capacity);
			sizeofElem = sizeof(hkReal);
			for( int i = 0; i < size; ++i )
			{
				static_cast<hkReal*>(dstArray.data)[i] = srcArray[i].asReal();
			}
			break;
		}

		case hkClassMember::TYPE_HALF:
		{
			dstArray.size = size;
			dstArray.capacity = dstArray.size;
			dstArray.data = hkMemHeapBufAlloc<hkHalf>(dstArray.capacity);
			sizeofElem = sizeof(hkHalf);
			for( int i = 0; i < size; ++i )
			{
				static_cast<hkHalf*>(dstArray.data)[i].setReal<false>(srcArray[i].asReal());
			}
			break;
		}

#define CASE_REAL_ARRAY(ENUM, TYPE) \
		case hkClassMember::ENUM: \
		{ \
			dstArray.size = size; \
			dstArray.capacity = dstArray.size; \
			dstArray.data = hkMemHeapBufAlloc<hk##TYPE>(dstArray.capacity); \
			sizeofElem = sizeof( hk##TYPE ); \
			for( int i = 0; i < size; ++i ) \
			{ \
				static_cast<hk##TYPE*>(dstArray.data)[i] = srcArray[i].as##TYPE(); \
			} \
		} \
		break

		CASE_REAL_ARRAY(TYPE_VECTOR4, Vector4);
		CASE_REAL_ARRAY(TYPE_QUATERNION, Quaternion);
		CASE_REAL_ARRAY(TYPE_MATRIX3, Matrix3);
		CASE_REAL_ARRAY(TYPE_ROTATION, Rotation);
		CASE_REAL_ARRAY(TYPE_QSTRANSFORM, QsTransform);
		CASE_REAL_ARRAY(TYPE_MATRIX4, Matrix4);
		CASE_REAL_ARRAY(TYPE_TRANSFORM, Transform);

#undef CASE_REAL_ARRAY

		case hkClassMember::TYPE_STRUCT:
		{		
			if( const hkClass* klass = getMemberClassAndCheck(member, srcArray[0].asObject() ) )
			{
				dstArray.size = size;
				int totalSizeInBytes = dstArray.size*klass->getObjectSize();
				dstArray.data = hkMemHeapBufAlloc<char>(totalSizeInBytes);
				dstArray.capacity = totalSizeInBytes / klass->getObjectSize();
				sizeofElem = klass->getObjectSize();
				hkString::memSet(dstArray.data, 0, srcArray.getSize()*klass->getObjectSize());
				for( int i = 0; i < size && result == HK_SUCCESS; ++i )
				{
					HK_ASSERT(0x1cae34b5, hkString::strCmp(member.getClass()->getName(), srcArray[i].asObject().getClass().getName()) == 0);
					void* ptr = hkAddByteOffset( dstArray.data, i*klass->getObjectSize() );
					result = fillNativeMembers( ptr, srcArray[i].asObject(), copyInfoOut );
				}
			}
			break;
		}
		case hkClassMember::TYPE_POINTER:
		{
			//if( const hkClass* klass = getMemberClassAndCheck(member, srcArray[0].asObject() ) )
			{
				dstArray.size = size;
				dstArray.capacity = dstArray.size;
				dstArray.data = hkMemHeapBufAlloc<void*>(dstArray.capacity);
				sizeofElem = hkSizeOf(void*);
				hkString::memSet(dstArray.data, 0, dstArray.size*sizeofElem);
				for( int i = 0; i < size; ++i )
				{
					hkDataObject o = srcArray[i].asObject();
					copyInfoOut.addPointer( o, static_cast<void**>(dstArray.data)+i, owningReference );
				}
			}
			break;
		}
		case hkClassMember::TYPE_VARIANT:
		{
			HK_ASSERT(0x58b5a9b9, !m_allocatedOnHeap);
			dstArray.size = size;
			dstArray.capacity = dstArray.size;
			dstArray.data = hkMemHeapBufAlloc<hkVariant>(dstArray.capacity);
			sizeofElem = hkSizeOf(hkVariant);
			hkString::memSet(dstArray.data, 0, dstArray.size*sizeofElem);
			for( int i = 0; i < size; ++i )
			{
				hkDataObject o = srcArray[i].asObject();
				copyInfoOut.addVariant( o, static_cast<hkVariant*>(dstArray.data)+i, owningReference );
			}
			break;
		}
		case hkClassMember::TYPE_CSTRING:
		{
			HK_ASSERT(0x36696467, !m_allocatedOnHeap);
			dstArray.size = size;
			dstArray.capacity = dstArray.size;
			dstArray.data = hkMemHeapBufAlloc<char*>(dstArray.capacity);
			sizeofElem = hkSizeOf(char*);
			for( int i = 0; i < size && result == HK_SUCCESS; ++i )
			{
				result = fillNativeSimpleMember(hkAddByteOffset(dstArray.data, sizeofElem*i), member.getSubType(), srcArray[i], HK_NULL, owningReference, copyInfoOut );
			}
			break;
		}
		case hkClassMember::TYPE_STRINGPTR:
		{
			dstArray.size = size;
			dstArray.capacity = dstArray.size;
			dstArray.data = hkMemHeapBufAlloc<hkStringPtr>(dstArray.capacity);
			sizeofElem = hkSizeOf(hkStringPtr);
			for( int i = 0; i < size && result == HK_SUCCESS; ++i )
			{
				result = fillNativeSimpleMember(hkAddByteOffset(dstArray.data, sizeofElem*i), member.getSubType(), srcArray[i], HK_NULL, owningReference, copyInfoOut );
			}
			break;
		}
		case hkClassMember::TYPE_VOID:
		{
			break;
		}
		default:
		{
			HK_ASSERT3(0x72c24629, false, "Unexpected type " << member.getSubType() << " found.");
			return HK_FAILURE;
		}
	}

	HK_ASSERT2(0x667acc1d, (sizeofElem > 0) == (dstArray.data!=HK_NULL), "internal consistency error" );

	if( sizeofElem > 0 )
	{
		DummyArray& dst = *static_cast<DummyArray*>(address);
		dst.data = dstArray.data;
		dst.size = dstArray.size;
		copyInfoOut.addAlloc( dstArray.data, sizeofElem * dstArray.size );
		LOG(("0x%p Allocate hkArray/hkSimpleArray %i elems, %i bytes", dstArray.data, dstArray.size, sizeofElem * dstArray.size));
	}
	// Not required for hkRelArray as there is no extra allocation
	if( member.getType() == hkClassMember::TYPE_ARRAY )
	{
		if( !m_allocatedOnHeap )
		{
			dstArray.capacity |= hkArray<char>::DONT_DEALLOCATE_FLAG;
		}
		DummyArray& dst = *static_cast<DummyArray*>(address);
		dst.capacity = dstArray.capacity;
	}
	return result;
}

hkResult hkDataObjectToNative::copyIntoRelArray( void* address, const hkClassMember& member, const hkDataArray& srcArray, CopyInfoOut& copyInfoOut, void *& relArrayAddress, void* objectStart )
{
	HK_ASSERT( 0x302c7fc9, !m_allocatedOnHeap || member.getType() != hkClassMember::TYPE_SIMPLEARRAY ); // packfile data support only

	DummyRelArray dstArray = { 0, 0 };
	if( srcArray.getSize() == 0 )
	{
		*static_cast<DummyRelArray*>(address) = dstArray;
		return HK_SUCCESS;
	}

	hkResult result = HK_SUCCESS;
	int sizeofElem = 0;
	hkBool32 owningReference = member.getFlags().get(hkClassMember::NOT_OWNED) == 0;

	switch( member.getSubType() )
	{
	case hkClassMember::TYPE_BOOL:
		{
			dstArray.size = hkUint16(srcArray.getSize());
			sizeofElem = sizeof(hkBool);
			for( int i = 0; i < srcArray.getSize(); ++i )
			{
				int val = srcArray[i].asInt();
				static_cast<hkBool*>(relArrayAddress)[i] = (val != 0);
			}
			break;
		}
#define CASE_INTEGER(ENUM, TYPE) \
	case hkClassMember::ENUM: \
		{ \
		dstArray.size = hkUint16(srcArray.getSize()); \
		sizeofElem = sizeof(TYPE); \
		for( int i = 0; i < srcArray.getSize(); ++i ) \
		{ \
		static_cast<TYPE*>(relArrayAddress)[i] = static_cast<TYPE>(srcArray[i].asInt()); \
		} \
		} \
		break

#define CASE_64BIT_INTEGER(ENUM, TYPE) \
	case hkClassMember::ENUM: \
		{ \
		dstArray.size = hkUint16(srcArray.getSize()); \
		sizeofElem = sizeof(TYPE); \
		for( int i = 0; i < srcArray.getSize(); ++i ) \
		{ \
		static_cast<TYPE*>(relArrayAddress)[i] = static_cast<TYPE>(srcArray[i].asInt64()); \
		} \
		} \
		break


		CASE_INTEGER(TYPE_CHAR, char);
		CASE_INTEGER(TYPE_INT8, hkInt8);
		CASE_INTEGER(TYPE_UINT8, hkUint8);
		CASE_INTEGER(TYPE_INT16, hkInt16);
		CASE_INTEGER(TYPE_UINT16, hkUint16);
		CASE_INTEGER(TYPE_INT32, hkInt32);
		CASE_INTEGER(TYPE_UINT32, hkUint32);

		CASE_64BIT_INTEGER(TYPE_INT64, hkInt64);
		CASE_64BIT_INTEGER(TYPE_UINT64, hkUint64);
		CASE_64BIT_INTEGER(TYPE_ULONG, hkUlong);
#undef CASE_INTEGER
#undef CASE_64BIT_INTEGER

	case hkClassMember::TYPE_REAL:
		{
			dstArray.size = hkUint16(srcArray.getSize());
			sizeofElem = sizeof(hkReal);
			for( int i = 0; i < srcArray.getSize(); ++i )
			{
				static_cast<hkReal*>(relArrayAddress)[i] = srcArray[i].asReal();
			}
			break;
		}

	case hkClassMember::TYPE_HALF:
		{
			dstArray.size = hkUint16(srcArray.getSize());
			sizeofElem = sizeof(hkHalf);
			for( int i = 0; i < srcArray.getSize(); ++i )
			{
				static_cast<hkHalf*>(relArrayAddress)[i].setReal<false>(srcArray[i].asReal());
			}
			break;
		}

#define CASE_REAL_ARRAY(ENUM, TYPE) \
	case hkClassMember::ENUM: \
		{ \
		dstArray.size = hkUint16(srcArray.getSize()); \
		sizeofElem = sizeof( hk##TYPE ); \
		for( int i = 0; i < srcArray.getSize(); ++i ) \
		{ \
		static_cast<hk##TYPE*>(relArrayAddress)[i] = srcArray[i].as##TYPE(); \
		} \
		} \
		break

		CASE_REAL_ARRAY(TYPE_VECTOR4, Vector4);
		CASE_REAL_ARRAY(TYPE_QUATERNION, Quaternion);
		CASE_REAL_ARRAY(TYPE_MATRIX3, Matrix3);
		CASE_REAL_ARRAY(TYPE_ROTATION, Rotation);
		CASE_REAL_ARRAY(TYPE_QSTRANSFORM, QsTransform);
		CASE_REAL_ARRAY(TYPE_MATRIX4, Matrix4);
		CASE_REAL_ARRAY(TYPE_TRANSFORM, Transform);

#undef CASE_REAL_ARRAY

	case hkClassMember::TYPE_STRUCT:
		{
			if( const hkClass* klass = getMemberClassAndCheck(member, srcArray[0].asObject() ) )
			{
				dstArray.size = hkUint16(srcArray.getSize());
				sizeofElem = klass->getObjectSize();
				hkString::memSet(relArrayAddress, 0, srcArray.getSize()*klass->getObjectSize());
				for( int i = 0; i < srcArray.getSize() && result == HK_SUCCESS; ++i )
				{
					HK_ASSERT(0x1cae34b5, hkString::strCmp(member.getClass()->getName(), srcArray[i].asObject().getClass().getName()) == 0);
					void* ptr = hkAddByteOffset( relArrayAddress, i*klass->getObjectSize() );
					result = fillNativeMembers( ptr, srcArray[i].asObject(), copyInfoOut );
				}
			}
			break;
		}
	case hkClassMember::TYPE_POINTER:
		{
			//if( const hkClass* klass = getMemberClassAndCheck(member, srcArray[0].asObject() ) )
			{
				dstArray.size = hkUint16(srcArray.getSize());
				sizeofElem = hkSizeOf(void*);
				hkString::memSet(relArrayAddress, 0, dstArray.size*sizeofElem);
				for( int i = 0; i < srcArray.getSize(); ++i )
				{
					hkDataObject o = srcArray[i].asObject();
					copyInfoOut.addPointer( o, static_cast<void**>(relArrayAddress)+i, owningReference );
				}
			}
			break;
		}
	case hkClassMember::TYPE_VARIANT:
		{
			HK_ASSERT(0x58b5a9b9, !m_allocatedOnHeap);
			dstArray.size = hkUint16(srcArray.getSize());
			sizeofElem = hkSizeOf(hkVariant);
			hkString::memSet(relArrayAddress, 0, dstArray.size*sizeofElem);
			for( int i = 0; i < srcArray.getSize(); ++i )
			{
				hkDataObject o = srcArray[i].asObject();
				copyInfoOut.addVariant( o, static_cast<hkVariant*>(relArrayAddress)+i, owningReference );
			}
			break;
		}
	case hkClassMember::TYPE_CSTRING:
		{
			HK_ASSERT(0x36696467, !m_allocatedOnHeap);
			dstArray.size = hkUint16(srcArray.getSize());
			sizeofElem = hkSizeOf(char*);
			for( int i = 0; i < srcArray.getSize() && result == HK_SUCCESS; ++i )
			{
				result = fillNativeSimpleMember(hkAddByteOffset(relArrayAddress, sizeofElem*i), member.getSubType(), srcArray[i], HK_NULL, owningReference, copyInfoOut );
			}
			break;
		}
	case hkClassMember::TYPE_STRINGPTR:
		{
			dstArray.size = hkUint16(srcArray.getSize());
			sizeofElem = hkSizeOf(hkStringPtr);
			for( int i = 0; i < srcArray.getSize() && result == HK_SUCCESS; ++i )
			{
				result = fillNativeSimpleMember(hkAddByteOffset(relArrayAddress, sizeofElem*i), member.getSubType(), srcArray[i], HK_NULL, owningReference, copyInfoOut );
			}
			break;
		}
	case hkClassMember::TYPE_VOID:
		{
			break;
		}
	default:
		{
			HK_ASSERT3(0x72c24629, false, "Unexpected type " << member.getSubType() << " found.");
			return HK_FAILURE;
		}
	}

	if( sizeofElem > 0 )
	{
		DummyRelArray& dst = *static_cast<DummyRelArray*>(address);
		HK_ASSERT2(0x3ece76ae, (static_cast<char*>(relArrayAddress) - static_cast<char*>(objectStart)) < 65536, "Pointer difference too large for hkRelArray");
		dst.offset = hkUint16(static_cast<char*>(relArrayAddress) - static_cast<char*>(address));
		relArrayAddress = hkAddByteOffset(relArrayAddress, HK_NEXT_MULTIPLE_OF(hkRelArrayUtil::RELARRAY_ALIGNMENT, dstArray.size * sizeofElem));
		dst.size = dstArray.size;
	}
	return result;
}


hkResult hkDataObjectToNative::fillNativeEnumMember
	( void* address
	, const hkClassMember& member
	, const hkDataObject::Value& value
	, const hkClass& klass )
{
	switch( member.getType() )
	{
		case hkClassMember::TYPE_ENUM:
		{
			if( member.hasEnumClass() )
			{
				//const hkClassEnum& e = member.getEnumClass();
				int enumval = value.asInt();
				member.setEnumValue(address, enumval);
			}
			break;
		}
		case hkClassMember::TYPE_FLAGS:
		{
			if( member.hasEnumClass() )
			{
				//const hkClassEnum& e = member.getEnumClass();
				int flags = value.asInt();
				member.setEnumValue(address, flags);
			}
			break;
		}
		default:
		{
			HK_ASSERT3(0x69e21810, false, "Unexpected type " << member.getType() << " found.");
			return HK_FAILURE;
		}
	}
	return HK_SUCCESS;
}

template< typename Value >
hkResult hkDataObjectToNative::fillNativeSimpleMember
	( void* address
	, hkClassMember::Type mtype
	, const Value& value
	, const hkClass* structClass
	, hkBool32 owningReference
	, CopyInfoOut& copyInfoOut )
{
	switch( mtype )
	{
		case hkClassMember::TYPE_BOOL:
		{
			*static_cast<hkBool*>( address ) = (value.asInt() != 0);
			break;
		}
#define CASE_INTEGER(ENUM, TYPE) \
		case hkClassMember::ENUM: *static_cast<TYPE*>( address ) = static_cast<TYPE>( value.asInt() ); \
			break

		CASE_INTEGER(TYPE_CHAR, char);
		CASE_INTEGER(TYPE_INT8, hkInt8);
		CASE_INTEGER(TYPE_UINT8, hkUint8);
		CASE_INTEGER(TYPE_INT16, hkInt16);
		CASE_INTEGER(TYPE_UINT16, hkUint16);
		CASE_INTEGER(TYPE_INT32, hkInt32);
		CASE_INTEGER(TYPE_UINT32, hkUint32);

#undef CASE_INTEGER

#define CASE_64BIT_INTEGER(ENUM, TYPE) \
		case hkClassMember::ENUM: \
		{ \
			*static_cast<TYPE*>( address ) = static_cast<TYPE>( value.asInt64() ); \
			break; \
		}
		
		CASE_64BIT_INTEGER(TYPE_INT64, hkInt64);
		CASE_64BIT_INTEGER(TYPE_UINT64, hkUint64);
		CASE_64BIT_INTEGER(TYPE_ULONG, hkUlong);
#undef CASE_64BIT_INTEGER

		case hkClassMember::TYPE_REAL:
		{
			*static_cast<hkReal*>( address ) = value.asReal();
			break;
		}
		case hkClassMember::TYPE_HALF:
		{
			static_cast<hkHalf*>( address )->setReal<false>(value.asReal());
			break;
		}
		case hkClassMember::TYPE_POINTER:
		{
			if( structClass ) //const hkClass* klass = getMemberClassAndCheck(member, srcValue.asObject(), classReg)
			{
				hkDataObject o = value.asObject();
				copyInfoOut.addPointer( o, static_cast<void**>(address), owningReference);
			}
			break;
		}
		case hkClassMember::TYPE_STRUCT:
		{
			if( structClass )
			{
				if( fillNativeMembers( address, value.asObject(), copyInfoOut ) == HK_FAILURE )
				{
					return HK_FAILURE;
				}
			}
			break;
		}

#define CASE_REAL_ARRAY(ENUM, TYPE) \
		case hkClassMember::ENUM: \
		{ \
			*static_cast<hk##TYPE*>(address) = value.as##TYPE(); \
			break; \
		}

		CASE_REAL_ARRAY(TYPE_VECTOR4, Vector4);
		CASE_REAL_ARRAY(TYPE_QUATERNION, Quaternion);
		CASE_REAL_ARRAY(TYPE_MATRIX3, Matrix3);
		CASE_REAL_ARRAY(TYPE_ROTATION, Rotation);
		CASE_REAL_ARRAY(TYPE_QSTRANSFORM, QsTransform);
		CASE_REAL_ARRAY(TYPE_MATRIX4, Matrix4);
		CASE_REAL_ARRAY(TYPE_TRANSFORM, Transform);

#undef CASE_REAL_ARRAY

		case hkClassMember::TYPE_CSTRING:
		{
			if( m_allocatedOnHeap ) // packfile data support only
			{
				return HK_FAILURE;
			}
			char* s = value.asString() ? hkString::strDup(value.asString()) : HK_NULL;
			if( s )
			{
				LOG(("0x%p Allocate string \"%s\"", s, s));
				copyInfoOut.addAlloc( s, -1 );
			}
			*static_cast<const char**>(address) = s;
			break;
		}
		case hkClassMember::TYPE_STRINGPTR:
		{
			HK_COMPILE_TIME_ASSERT(hkSizeOf(hkStringPtr) == hkSizeOf(char*));
			char* s = value.asString() ? hkString::strDup(value.asString()) : HK_NULL;
			if( s )
			{
				LOG(("0x%p Allocate string \"%s\"", s, s));
				if( m_allocatedOnHeap ) // stringptr destructor takes care of it
				{
					HK_COMPILE_TIME_ASSERT( hkStringPtr::OWNED_FLAG == 1 );
					s += 1;
				}
				else // packfiledata is responsible for it
				{
					copyInfoOut.addAlloc(s, -1);
				}
			}
			*static_cast<const char**>(address) = s;
			break;
		}
		case hkClassMember::TYPE_VARIANT:
		{
			if( m_allocatedOnHeap ) // packfile data support only
			{
				return HK_FAILURE;
			}
			hkDataObject o = value.asObject();
			copyInfoOut.addVariant( o, static_cast<hkVariant*>(address), owningReference);
			break;
		}
		case hkClassMember::TYPE_ENUM:
		case hkClassMember::TYPE_FLAGS:
		case hkClassMember::TYPE_ARRAY:
		case hkClassMember::TYPE_SIMPLEARRAY:
		case hkClassMember::TYPE_HOMOGENEOUSARRAY:
		case hkClassMember::TYPE_FUNCTIONPOINTER:
		case hkClassMember::TYPE_INPLACEARRAY:
		default:
		{
			HK_ASSERT3(0x178f953c, false, "Unexpected type " << mtype << " found.");
			return HK_FAILURE;
		}
	}
	return HK_SUCCESS;
}

hkResult hkDataObjectToNative::fillNativeMembers( void* dstObject, const hkDataObject& srcObj, CopyInfoOut& copyInfoOut )
{
	HK_ASSERT(0x44b6ca12, !srcObj.isNull());
	hkResult result = HK_SUCCESS;
	const hkClass* curClass = findClassOf(srcObj);

	void* relArrayAddress = hkAddByteOffset(dstObject, HK_NEXT_MULTIPLE_OF(hkRelArrayUtil::RELARRAY_ALIGNMENT, curClass->getObjectSize()));
	while( curClass ) 
	{
		for( int memberIndex = 0; memberIndex < curClass->getNumDeclaredMembers(); ++memberIndex )
		{
			const hkClassMember& member = curClass->getDeclaredMember(memberIndex);
			void* address = hkAddByteOffset(dstObject, member.getOffset() );

			hkClassMember::Type dstType = member.getType();

			const hkDataObject::Value srcValue = const_cast<hkDataObject&>(srcObj)[member.getName()];

			// srcObj may not have a member if there is a default but we still need to assign a value
			if( !srcValue.isSet())
			{
				hkDataClass::MemberInfo mInfo;
		
				if (srcValue.isValid())
				{
					srcValue.getMemberInfo(mInfo);
				}
				if( member.getType() == hkClassMember::TYPE_ARRAY ) // init the array capacity
				{
					static_cast<DummyArray*>(address)->capacity = hkArray<char>::DONT_DEALLOCATE_FLAG;
					// Can't assume we can continue, array of structs may still need defaults
					//continue;
				}

				if(curClass->hasDeclaredDefault(memberIndex) && srcObj.getImplementation()->createdDuringPatching())
				{
					hkString::memCpy(address, curClass->getDeclaredDefault(memberIndex), member.getSizeInBytes());
					continue;
				}
				
				if(srcValue.isValid() && mInfo.m_valuePtr)
				{
					// Don't continue if we have a default value
				}
				// Don't early out for structs as they could have embedded members with defaults
				else if((dstType != hkClassMember::TYPE_STRUCT) || (member.getFlags().get(hkClassMember::SERIALIZE_IGNORED)))
 				{
 					continue;
 				}
			}

			// todo: move nearer usage
			int numelem;
			int elemsize;
			if (member.getCstyleArraySize())
			{
				numelem = member.getCstyleArraySize();
				elemsize = member.getSizeInBytes()/numelem; 
			}
			else
			{
				numelem = 1;
				elemsize = member.getSizeInBytes();
			}
			
			hkBool32 owningReference = member.getFlags().get(hkClassMember::NOT_OWNED) == 0;

			switch( dstType )
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

				case hkClassMember::TYPE_CSTRING:
				case hkClassMember::TYPE_STRINGPTR:

				case hkClassMember::TYPE_STRUCT:
				case hkClassMember::TYPE_VARIANT:
				{
					if( numelem == 1 )
					{
						result = fillNativeSimpleMember( address, dstType, srcValue, member.getClass(), owningReference, copyInfoOut );
					}
					else
					{
						hkDataArray a = srcValue.asArray();
						HK_ASSERT(0x7016d093, a.getSize() <= numelem);
						for( int i = 0; i < a.getSize() && result == HK_SUCCESS; ++i )
						{
							result = fillNativeSimpleMember( hkAddByteOffset(address, i*elemsize), dstType, a[i], member.getClass(), owningReference, copyInfoOut );
						}
					}
					break;
				}
				case hkClassMember::TYPE_ARRAY:
				case hkClassMember::TYPE_SIMPLEARRAY:
				{
					if( m_allocatedOnHeap &&
						(member.getType() == hkClassMember::TYPE_SIMPLEARRAY
						|| member.getSubType() == hkClassMember::TYPE_VARIANT
						|| member.getSubType() == hkClassMember::TYPE_CSTRING) ) // packfile data support only
					{
						result = HK_FAILURE;
					}
					else
					{
						result = copyIntoNativeArray( address, member, srcValue.asArray(), copyInfoOut);
					}
					break;
				}
				case hkClassMember::TYPE_RELARRAY:
				{
					if( m_allocatedOnHeap &&
						(member.getType() == hkClassMember::TYPE_SIMPLEARRAY
						|| member.getSubType() == hkClassMember::TYPE_VARIANT
						|| member.getSubType() == hkClassMember::TYPE_CSTRING) ) // packfile data support only
					{
						result = HK_FAILURE;
					}
					else
					{
						result = copyIntoRelArray( address, member, srcValue.asArray(), copyInfoOut, relArrayAddress, dstObject);
					}
					break;
				}
				case hkClassMember::TYPE_POINTER:
				{
					if( member.getSubType() == hkClassMember::TYPE_CHAR )
					{
						if( numelem == 1 )
						{
							result = fillNativeSimpleMember( address, hkClassMember::TYPE_CSTRING, srcValue, HK_NULL, owningReference, copyInfoOut );
						}
						else
						{
							hkDataArray a = srcValue.asArray();
							HK_ASSERT(0x726b7ceb, a.getSize() <= numelem);
							for( int i = 0; i < a.getSize() && result == HK_SUCCESS; ++i )
							{
								result = fillNativeSimpleMember( hkAddByteOffset(address, i*elemsize), hkClassMember::TYPE_CSTRING, a[i], HK_NULL, owningReference, copyInfoOut );
							}
						}
					}
					else if( member.getSubType() == hkClassMember::TYPE_STRUCT )
					{
						if( numelem == 1 )
						{
							result = fillNativeSimpleMember( address, dstType, srcValue, getMemberClassAndCheck(member, srcValue.asObject() ), owningReference, copyInfoOut );
						}
						else
						{
							hkDataArray a = srcValue.asArray();
							HK_ASSERT(0x50eb8bb5, a.getSize() <= numelem);
							for( int i = 0; i < a.getSize() && result == HK_SUCCESS; ++i )
							{
								result = fillNativeSimpleMember( hkAddByteOffset(address, i*elemsize), dstType, a[i], getMemberClassAndCheck(member, a[i].asObject() ), owningReference, copyInfoOut );
							}
						}
					}
					break;
				}
				case hkClassMember::TYPE_ENUM:
				case hkClassMember::TYPE_FLAGS:
				{
					result = fillNativeEnumMember( address, member, srcValue, *curClass );
					break;
				}
				case hkClassMember::TYPE_HOMOGENEOUSARRAY:
				{
					if( m_allocatedOnHeap ) // packfile data support only
					{
						result = HK_FAILURE;
					}
					else
					{
						HK_ASSERT2(0x7edfac51, numelem == 1, "Multidimentional arrays are not supported.");
						struct DummyHomogeneousArray
						{
							const hkClass* klass;
							void* data;
							int size;
						};
						DummyHomogeneousArray& dstArray = *static_cast<DummyHomogeneousArray*>(address);
						hkDataArray srcArray = srcValue.asArray();
						dstArray.klass = getMemberClassAndCheck(member, srcArray[0].asObject() );
						dstArray.size = srcArray.getSize();
						int objectSize = dstArray.klass->getObjectSize();
						int totalSizeInBytes = objectSize*dstArray.size;
						dstArray.data = hkMemHeapBufAlloc<char>(totalSizeInBytes);
						hkString::memSet(dstArray.data, 0, totalSizeInBytes);
						LOG(("0x%p Allocate hkHomogeneousArray %d elems, %i bytes", dstArray.data, dstArray.size, totalSizeInBytes));
						copyInfoOut.addAlloc( dstArray.data, objectSize*dstArray.size );
						for( int i = 0; i < dstArray.size && result == HK_SUCCESS; ++i )
						{
							result = fillNativeMembers( hkAddByteOffset(dstArray.data, objectSize*i), srcArray[i].asObject(), copyInfoOut );
						}
					}
					break;
				}
				case hkClassMember::TYPE_FUNCTIONPOINTER:
				case hkClassMember::TYPE_INPLACEARRAY:
				default:
				{
					HK_ASSERT(0x28cf84ad, 0);
					result = HK_FAILURE;
				}
			}
			if( result == HK_FAILURE )
			{
				HK_ON_DEBUG(char typeName[256]);
				//HK_ASSERT3(0x43ae1f6b, false, "Cannot generate 'alive' object of class containing member " << curClass->getName() << "::" << member.getName() << " of type " << getMemberTypeName(member, typeName, hkSizeOf(typeName)) << ".");
				HK_WARN(0x43ae1f6b, "Cannot generate 'alive' object of class containing member " << curClass->getName() << "::" << member.getName() << " of type " << getMemberTypeName(member, typeName, hkSizeOf(typeName)) << ".");
				return HK_FAILURE;
			}
		}
		curClass = curClass->getParent();
	}
	return result;
}

const hkClass* hkDataObjectToNative::getMemberClassAndCheck(const hkClassMember& member, const hkDataObject& obj )
{
	if( obj.isNull() )
	{
		return HK_NULL;
	}
	const hkClass* expectedClass = member.getClass();
	const char* actualClassName = obj.getClass().getName();
	const hkClass* actualClass = actualClassName ? m_classReg->getClassByName(actualClassName) : HK_NULL;
	if( expectedClass == HK_NULL ) // native is void*, so it's compatible
	{
		return actualClass;
	}
	else if( actualClass == HK_NULL ) // don't know about this class
	{
		return HK_NULL;
	}
	if( hkString::strCmp(expectedClass->getName(), actualClassName) != 0 )
	{
		// didn't match exactly, so check if the actualClass is derived from what we expect
		if( expectedClass->isSuperClass(*actualClass) )
		{
			return actualClass;
		}
		else if( const hkVariant* override = member.getAttribute("hk.DataObjectType") )
		{
			// Find if it has a override type to handle strange pseudo-unions like hkpEntity::m_motion 
			// whose hkClass is hkpMaxSizeMotion but is actually a hkpMotion.
			const char* expectedName = static_cast<hkDataObjectTypeAttribute*>(override->m_object)->m_typeName;
			expectedClass = m_classReg->getClassByName(expectedName);
			if( expectedClass->isSuperClass(*actualClass) )
			{
				return actualClass;
			}
		}
		HK_ASSERT3(0x431c29ce, false, "The Data object " << actualClassName << " differs from hkClass " << expectedClass->getName());
	}
	return expectedClass;
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
