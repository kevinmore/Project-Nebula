/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Copier/hkObjectCopier.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Container/RelArray/hkRelArray.h>
#include <Common/Base/Container/RelArray/hkRelArrayUtil.h>
#include <Common/Base/System/Io/OArchive/hkOArchive.h>
#include <Common/Base/System/Io/Writer/hkStreamWriter.h>
#include <Common/Base/System/Io/Writer/Array/hkArrayStreamWriter.h>
#include <Common/Serialize/Serialize/hkRelocationInfo.h>

#if 0
	extern "C" int printf(const char*,...);
#	define PRINT(A) printf A
#else
#	define PRINT(A) /* nothing */
#endif

namespace
{
	bool inRange(int num, int lo, int hi)
	{
		return num >= lo && num < hi;
	}
	template <typename T>
	T min2(T a, T b)
	{
		return a<b ? a : b;
	}
}
extern const hkClass hkClassClass;

hkObjectCopier::hkObjectCopier(const hkStructureLayout& layoutIn, const hkStructureLayout& layoutOut, ObjectCopierFlags flags)
	:	m_layoutIn( layoutIn ), m_layoutOut( layoutOut ), m_flags( flags )
{
	m_byteSwap = ( m_layoutIn.getRules().m_littleEndian != m_layoutOut.getRules().m_littleEndian );
	HK_ASSERT(0x3bcc68e5, layoutIn.getRules().m_bytesInPointer == sizeof(void*) );
}

hkObjectCopier::~hkObjectCopier()
{
}

namespace 
{
	typedef hkArray<char> hkAnyArray;

	template <typename T>
	struct ObjectCopier_DummyArray
	{
		T* data;
		int size;
		int capAndFlags;
	};

	struct ObjectCopier_DummyHomogeneousArray
	{
		hkClass* klass;
		void* data;
		int size;
		//		int capAndFlags;
	};
}

static void objectCopierPadUp( hkStreamWriter* w, int pad=HK_REAL_ALIGNMENT )
{
	int o = w->tell();
	hkLocalArray<char> buf(pad);
	const unsigned char padChar = HK_ON_DEBUG( (unsigned char)(0x7f) ) + 0; 
	buf.setSize(pad, (const char)padChar);
	if( o & (pad-1) )
	{
		w->write( buf.begin(), pad - (o&(pad-1)) );
	}
}

static int objectCopier_calcCArraySize( const hkClassMember& member )
{
	return (member.getCstyleArraySize()) ? member.getCstyleArraySize() : 1;
}

static int writeZeros(hkOArchive& oa, int numZeros, hkLocalArray<char>& zeroArray)
{
	zeroArray.setSize(numZeros, 0);
	return oa.writeRaw(zeroArray.begin(), numZeros);
}

void hkObjectCopier::writeZero(
	hkOArchive& oa,
	const hkClassMember& memberOut,
	hkLocalArray<char>& zeroArray )
{
	//
	// Write zero for every different subtype
	//
	int size = 0;
	switch( memberOut.getType() )
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
		case hkClassMember::TYPE_HALF:
		case hkClassMember::TYPE_VECTOR4:
		case hkClassMember::TYPE_QUATERNION:
		case hkClassMember::TYPE_MATRIX3:
		case hkClassMember::TYPE_ROTATION:
		case hkClassMember::TYPE_QSTRANSFORM:
		case hkClassMember::TYPE_MATRIX4:
		case hkClassMember::TYPE_TRANSFORM:
		case hkClassMember::TYPE_ENUM:
		case hkClassMember::TYPE_FLAGS:
		{	
			size = memberOut.getSizeInBytes();
			break;
		}
		case hkClassMember::TYPE_ULONG:
		case hkClassMember::TYPE_POINTER:
		case hkClassMember::TYPE_FUNCTIONPOINTER:
		case hkClassMember::TYPE_STRINGPTR:
		{
			size = m_layoutOut.getRules().m_bytesInPointer * objectCopier_calcCArraySize( memberOut );
			break;
		}
		case hkClassMember::TYPE_ARRAY:
		case hkClassMember::TYPE_INPLACEARRAY:
		case hkClassMember::TYPE_SIMPLEARRAY:
		case hkClassMember::TYPE_HOMOGENEOUSARRAY:
		{
			// Twice the size of pointers.
			HK_COMPILE_TIME_ASSERT( sizeof(void*) <= 8 );

			if( memberOut.getType() == hkClassMember::TYPE_HOMOGENEOUSARRAY )
			{
				size += m_layoutOut.getRules().m_bytesInPointer;
			}

			size += m_layoutOut.getRules().m_bytesInPointer;
			size += sizeof(hkUint32); // size
			if( memberOut.getType() == hkClassMember::TYPE_ARRAY )
			{
				// create enough zeros for a pointer (might be 64 bits) and an hkUint32
				char zero[ 8 + 4 ] = { 0 };
				oa.writeRaw( zero, size );
				oa.write32u( hkUint32(hkAnyArray::DONT_DEALLOCATE_FLAG) ); // capacity and flags
				return; // early out
			}
			break;
		}
		case hkClassMember::TYPE_STRUCT: // single struct
		{
			const hkClass& sclass = memberOut.getStructClass();
			size = sclass.getObjectSize() * objectCopier_calcCArraySize( memberOut );
			break;
		}
		case hkClassMember::TYPE_VARIANT:
		{
			size = m_layoutOut.getRules().m_bytesInPointer * 2 * objectCopier_calcCArraySize( memberOut );
			break;
		}
		default:
		{
			HK_ERROR(0x5ef4e5a4, "Unknown class member type found!" );
			break;
		}
	}
	HK_ASSERT2(0x5ef4e5a4, size != 0, "Incorrect size for class member calculated.");

	writeZeros(oa, size, zeroArray);
}

static void writeUlongArray(hkOArchive& oa, int elemsize, int nelem, const void* startAddress)
{
	if( sizeof(hkUlong) == elemsize ) // same as host
	{
		oa.writeArrayGeneric( startAddress, elemsize, nelem );
	}
	else // target ulong is different
	{
#		if HK_POINTER_SIZE == 4
			typedef hkUint64 TargetUlong;
#		else
			typedef hkUint32 TargetUlong;
			HK_WARN( 0x57d9c254, "hkClassMember::TYPE_ULONG: conversion from '64-bit' to '32-bit', possible loss of data." );
#		endif
		HK_ASSERT( 0x57d9c255, elemsize == sizeof(TargetUlong) );
		hkLocalBuffer<TargetUlong> targetUlongArray(nelem);
		const hkUlong* hostUlongArray = static_cast<const hkUlong*>(startAddress);
		for( int i = 0; i < nelem; ++i )
		{
			targetUlongArray[i] = static_cast<TargetUlong>(hostUlongArray[i]);
		}
		oa.writeArrayGeneric( targetUlongArray.begin(), elemsize, nelem );
	}
}

static void writePodArray(hkOArchive& oa, const hkClassMember::Type mtype,
						  int elemsize, int nelem, const void* startAddress )
{
	switch( mtype )
	{
		case hkClassMember::TYPE_VECTOR4:
		case hkClassMember::TYPE_QUATERNION:
		case hkClassMember::TYPE_ROTATION:
		case hkClassMember::TYPE_MATRIX3:
		case hkClassMember::TYPE_MATRIX4:
		case hkClassMember::TYPE_QSTRANSFORM:
		case hkClassMember::TYPE_TRANSFORM:
		{
			nelem *= (elemsize/ sizeof(hkReal) );
			elemsize = sizeof(hkReal);
			/* fall through */
		}
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
		case hkClassMember::TYPE_HALF:
		{
			oa.writeArrayGeneric( startAddress, elemsize, nelem );
			break;
		}
		case hkClassMember::TYPE_ULONG:
		case hkClassMember::TYPE_POINTER:
		case hkClassMember::TYPE_FUNCTIONPOINTER:
		case hkClassMember::TYPE_ARRAY:
		case hkClassMember::TYPE_INPLACEARRAY:
		case hkClassMember::TYPE_STRUCT:
		case hkClassMember::TYPE_SIMPLEARRAY:
		case hkClassMember::TYPE_HOMOGENEOUSARRAY:
		case hkClassMember::TYPE_VARIANT:
		{
			HK_ASSERT2(0x747e1e04, 0, "Write POD array called with non-pod type." );
		}
		default:
		{
			HK_ERROR(0x747e1e03, "Unknown class member found during write of plain data array.");
		}
	}
}

static hkResult applyDefaultsFor(const hkClass& klass, const hkClassMember& mem, int memberIndex, hkStreamWriter* writer)
{
	hkResult res = klass.getDefault( memberIndex, writer ); // not copied, try to get default
	if( mem.getType() == hkClassMember::TYPE_STRUCT ) // single struct
	{
		const hkClass* structOut = mem.getClass();
		if( structOut )
		{
			int nelem = objectCopier_calcCArraySize( mem );
			for( int i = 0; i < nelem; ++i )
			{
				for( int m = 0; m < structOut->getNumMembers(); ++m )
				{
					const hkClassMember& structMem = structOut->getMember(m);
					if( applyDefaultsFor(*structOut, structMem, m, writer) == HK_SUCCESS )
					{
						res = HK_SUCCESS; // at least one has succeeded
					}
				}
			}
		}
	}
	return res;
}

hkBool32 hkObjectCopier::areMembersCompatible(const hkClassMember& src, const hkClassMember& dst)
{
	return src.getType() == dst.getType() && (src.getSubType() == dst.getSubType() || src.getType() == hkClassMember::TYPE_ENUM);
}

namespace
{
	struct RelArrayFixup
	{
		const hkClassMember* memberIn;
		const hkClassMember* memberOut;
		int position;
		const void* pointer;
		hkUint16 size;
	};
}

void hkObjectCopier::saveBody(
	const void* dataIn, // data source
	const hkClass& klassIn, // class source
	hkOArchive& dataOut, // dest data
	const hkClass& klassOut, // class dest
	hkLocalArray<char>& zeroArray
)
{
	int bodyStart = dataOut.getStreamWriter()->tell();

	hkArray<struct RelArrayFixup>::Temp fixupRelArrays;

	for( int memberIdx = 0, memberNum = klassOut.getNumMembers(); memberIdx < memberNum; ++memberIdx )
	{
		const hkClassMember& memberOut = klassOut.getMember( memberIdx );
		HK_ASSERT2(0x7e43c55a, memberOut.getOffset() >= (dataOut.getStreamWriter()->tell() - bodyStart), "Overlapping members in ObjectCopier");
		writeZeros(dataOut, memberOut.getOffset() - (dataOut.getStreamWriter()->tell() - bodyStart), zeroArray);

		if( memberOut.getFlags().get(hkClassMember::SERIALIZE_IGNORED))
		{
			if( m_flags.get(FLAG_RESPECT_SERIALIZE_IGNORED) )
			{
				writeZero( dataOut, memberOut, zeroArray );
				continue;
			}
			else if( m_flags.get(FLAG_APPLY_DEFAULT_IF_SERIALIZE_IGNORED) )
			{
				applyDefaultsFor(klassOut, memberOut, memberIdx, dataOut.getStreamWriter());  // try to get default
				continue;
			}
		}

		if( const hkClassMember* memberInPtr = (&klassIn==&klassOut) ? &memberOut : klassIn.getMemberByName( memberOut.getName() ) )
		{
			const hkClassMember& memberIn = *memberInPtr;
			if( areMembersCompatible(memberIn, memberOut) )
			{
				const void* addressIn = static_cast<const char*>(dataIn) + memberIn.getOffset();

				switch( memberOut.getType() )
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
					case hkClassMember::TYPE_HALF:
					case hkClassMember::TYPE_VECTOR4:
					case hkClassMember::TYPE_QUATERNION:
					case hkClassMember::TYPE_MATRIX3:
					case hkClassMember::TYPE_ROTATION:
					case hkClassMember::TYPE_QSTRANSFORM:
					case hkClassMember::TYPE_MATRIX4:
					case hkClassMember::TYPE_TRANSFORM:
					{
						int nelemIn = objectCopier_calcCArraySize( memberIn );
						int nelemOut = objectCopier_calcCArraySize( memberOut );
						int nelem = min2(nelemIn, nelemOut);
						int realSize = memberOut.getSizeInBytes();
						writePodArray( dataOut, memberOut.getType(),
										realSize / nelemOut, nelem, addressIn );
						break;
					}
					case hkClassMember::TYPE_ULONG:
					{
						int nelemIn = objectCopier_calcCArraySize( memberIn );
						int nelemOut = objectCopier_calcCArraySize( memberOut );
						int nelem = min2(nelemIn, nelemOut);
						writeUlongArray( dataOut, m_layoutOut.getRules().m_bytesInPointer, nelem, addressIn );
						break;
					}
					case hkClassMember::TYPE_ENUM:
					{
						if ( !memberIn.hasEnumClass() || !memberOut.hasEnumClass() )
						{
							break;
						}

						int nelemIn = objectCopier_calcCArraySize( memberIn );
						int nelemOut = objectCopier_calcCArraySize( memberOut );
						int nelem = min2(nelemIn, nelemOut);
						const hkClassEnum& enumIn = memberIn.getEnumClass();
						const hkClassEnum& enumOut = memberOut.getEnumClass();
						int enumBytes = memberIn.getSizeInBytes() / nelemIn;
						const void* enumInPtr = addressIn;
						HK_ASSERT( 0xc585890e, nelem >= 0 );
						for( int i = 0; i < nelem; ++i )
						{
							int valueIn = memberIn.getEnumValue(enumInPtr);
							int valueOut = 0;
							const char* nameIn;
							if ( enumIn.getNameOfValue(valueIn, &nameIn) == HK_SUCCESS )
							{
								if ( enumOut.getValueOfName( nameIn, &valueOut ) == HK_SUCCESS )
								{
								}
								else
								{
									HK_WARN(0x337d3f12, "couldn't convert "<< klassOut.getName() << "::" << enumOut.getName() << " " << nameIn << " to a value");
								}
							}
							else if( valueIn != 0 )
							{
								HK_WARN(0x337d3f13, "couldn't convert "<< klassIn.getName() << "::" << enumIn.getName() << " "<<valueIn<<" to a name");
							}
							switch( enumBytes )
							{
								case 1: dataOut.write8( hkInt8(valueOut) ); break;
								case 2: dataOut.write16( hkInt16(valueOut) ); break;
								case 4: dataOut.write32( hkInt32(valueOut) ); break;
							}
							enumInPtr = hkAddByteOffsetConst(enumInPtr, enumBytes);
						}
						break;
					}
					case hkClassMember::TYPE_ZERO:
					{					
						HK_ASSERT2(0x5e995008, false, "hkClassMember::TYPE_ZERO deprecated.");
						break;
					}
					case hkClassMember::TYPE_CSTRING:
					case hkClassMember::TYPE_STRINGPTR:
					case hkClassMember::TYPE_POINTER:
					case hkClassMember::TYPE_FUNCTIONPOINTER:
					{
						int nelem = objectCopier_calcCArraySize( memberOut );
						HK_COMPILE_TIME_ASSERT( sizeof(void*) <= 8 );
						hkUint64 zero = 0;
						for( int j = 0; j < nelem; ++j )
						{
							dataOut.writeRaw( &zero, m_layoutOut.getRules().m_bytesInPointer );
						}
						break;
					}
					case hkClassMember::TYPE_ARRAY:
					case hkClassMember::TYPE_SIMPLEARRAY:
					case hkClassMember::TYPE_HOMOGENEOUSARRAY:
					{
						HK_COMPILE_TIME_ASSERT( sizeof(void*) <= 8 );
						int arraySize = -1;
						if( memberOut.getType() == hkClassMember::TYPE_HOMOGENEOUSARRAY )
						{
							hkUint64 aid = 0; // class pointer
							dataOut.writeArrayGeneric( &aid, m_layoutOut.getRules().m_bytesInPointer, 1 );
							if( static_cast<const ObjectCopier_DummyHomogeneousArray*>( addressIn )->klass )
							{
								arraySize = static_cast<const ObjectCopier_DummyHomogeneousArray*>( addressIn )->size;
							}
							else
							{
								HK_WARN_ALWAYS(0xabba5a55, "Can't copy homogeneous array. No hkClass for " << klassIn.getName() << "::" << memberIn.getName() << ".");
								arraySize = 0; // no class info, can't copy it
							}
						}
						else if( memberOut.getSubType() == hkClassMember::TYPE_VOID )
						{
							arraySize = 0; // nonreflected array, can't copy it
						}
						else
						{
							arraySize = static_cast<const ObjectCopier_DummyArray<char>*>( addressIn )->size;
						}
						HK_ASSERT( 0xc585890e, arraySize >= 0 );

						hkUint64 aid = 0; // data pointer
						dataOut.writeArrayGeneric( &aid, m_layoutOut.getRules().m_bytesInPointer, 1 );
						dataOut.write32( arraySize ); // size
						if( memberOut.getType() == hkClassMember::TYPE_ARRAY )
						{
							int capAndFlags = arraySize | hkAnyArray::DONT_DEALLOCATE_FLAG;
							dataOut.write32( capAndFlags );
						}
						break;
					}
					case hkClassMember::TYPE_RELARRAY:
					{
						const hkRelArray<char>* arr = static_cast<const hkRelArray<char>*>(addressIn);
						RelArrayFixup raf;
						raf.memberIn = memberInPtr;
						raf.memberOut = &memberOut;
						raf.position = dataOut.getStreamWriter()->tell() - bodyStart;
						raf.pointer = arr->begin();
						raf.size = arr->getSize();
						dataOut.write16u(0); // size will be written later as well
						dataOut.write16u(0); // needs to be fixed up later
						fixupRelArrays.pushBack(raf);
						break;
					}
					case hkClassMember::TYPE_STRUCT: // single struct
					{
						const hkClass* structIn = memberIn.getClass();
						const hkClass* structOut = memberOut.getClass();
						if( structIn && structOut )
						{
							int nelem = min2( objectCopier_calcCArraySize( memberIn ), objectCopier_calcCArraySize( memberOut ) );
							for( int i = 0 ; i < nelem; ++i)
							{
								const void* din = static_cast<const char*>(addressIn)+i*structIn->getObjectSize();
								saveBody( din, *structIn, dataOut, *structOut, zeroArray );
							}
						}
						
						break;
					}
					case hkClassMember::TYPE_VARIANT:
					{
						hkUint64 aid[2] = {0,0}; // data, class pointer
						int nelem = min2( objectCopier_calcCArraySize( memberIn ), objectCopier_calcCArraySize( memberOut ) );
						for( int i = 0; i < nelem; ++i )
						{
							dataOut.writeArrayGeneric( aid, m_layoutOut.getRules().m_bytesInPointer, 2 );
						}
						break;
					}
					case hkClassMember::TYPE_FLAGS:
					{
						if ( !memberIn.hasEnumClass() || !memberOut.hasEnumClass() )
						{
							break;
						}

						int nelemIn = objectCopier_calcCArraySize( memberIn );
						int nelemOut = objectCopier_calcCArraySize( memberOut );
						int nelem = min2(nelemIn, nelemOut);
						const hkClassEnum& enumIn = memberIn.getEnumClass();
						const hkClassEnum& enumOut = memberOut.getEnumClass();
						int flagBytes = memberIn.getSizeInBytes() / nelemIn;
						const void* flagInPtr = addressIn;
						HK_ASSERT( 0xc585890e, nelem >= 0 );
						for( int arrayIndex = 0; arrayIndex < nelem; ++arrayIndex )
						{
							int valueIn = memberIn.getEnumValue(flagInPtr);
							hkArray<const char*> bitNames;
							int oldBitsUnknown = 0;
							enumIn.decomposeFlags( valueIn, bitNames, oldBitsUnknown );
							int valueOut = 0;
							for( int i = 0; i < bitNames.getSize(); ++i )
							{
								int val;
								if( enumOut.getValueOfName( bitNames[i], &val ) == HK_SUCCESS )
								{
									valueOut |= val;
								}
								else
								{
									HK_WARN( 0x718f5bfd, "Flags "<< klassIn.getName() << "::" << memberIn.getName()
											<< "(" << bitNames[i] << ") no longer a valid flag.");
									oldBitsUnknown |= val;
								}
							}
							// try to carry across unknown bits which don't conflict
							if( oldBitsUnknown )
							{
								int newBitsUnknown = 0;
								enumOut.decomposeFlags( oldBitsUnknown, bitNames, newBitsUnknown );
								valueOut |= newBitsUnknown;
								HK_WARN( 0x718f5bfe, "Propagating unknown nonconflicting flag bits in " << klassIn.getName()
										<< "::" << memberIn.getName() << "("<< newBitsUnknown << ")" );
							}

							switch( flagBytes )
							{
								case 1: dataOut.write8( hkInt8(valueOut) ); break;
								case 2: dataOut.write16( hkInt16(valueOut) ); break;
								case 4: dataOut.write32( hkInt32(valueOut) ); break;
							}
							flagInPtr = hkAddByteOffsetConst(flagInPtr, flagBytes);
						}
						break;
					}
					case hkClassMember::TYPE_INPLACEARRAY:
					default:
					{
						HK_ERROR(0x641e3e03, "Unknown class member found during write of data.");
					}
				}
				continue; // successful copy
			}
		}
		applyDefaultsFor(klassOut, memberOut, memberIdx, dataOut.getStreamWriter()); // not copied, try to get default
	}

	// skip possible end padding
	HK_ASSERT2(0x7adc789f, klassOut.getObjectSize() >= (dataOut.getStreamWriter()->tell() - bodyStart), "Overlapping members in ObjectCopier");
	writeZeros(dataOut, klassOut.getObjectSize() - (dataOut.getStreamWriter()->tell() - bodyStart), zeroArray);

	for(int fixup=0;fixup<fixupRelArrays.getSize();fixup++)
	{
		const hkClassMember* memberIn = fixupRelArrays[fixup].memberIn;
		const hkClassMember* memberOut = fixupRelArrays[fixup].memberOut;
		const void* pointer = fixupRelArrays[fixup].pointer;
		hkUint16 size = fixupRelArrays[fixup].size;
		
		// Leave padding to always have RELARRAY_ALIGNMENT on hkRelArray data.
		// We assume that the object start is always aligned to RELARRAY_ALIGNMENT.
		objectCopierPadUp(dataOut.getStreamWriter(), hkRelArrayUtil::RELARRAY_ALIGNMENT);

		int relArrayPos = fixupRelArrays[fixup].position;
		int curPos = dataOut.getStreamWriter()->tell() - bodyStart;
		dataOut.getStreamWriter()->seek(relArrayPos, hkStreamWriter::STREAM_SET);
		dataOut.write16u(size);
		dataOut.write16u(static_cast<hkUint16>(curPos-relArrayPos));
		dataOut.getStreamWriter()->seek(0, hkStreamWriter::STREAM_END);

		// Need to copy out size, member.getType()'s from pointer
		if( size )
		{
			if( memberOut->getSubType() == hkClassMember::TYPE_VOID )
			{
			}
			else if( memberOut->getSubType() == hkClassMember::TYPE_POINTER ) // array of pointer
			{
				HK_ASSERT2(0x4765fb5e, 0, "Type not supported by hkRelArray");
			}
			else if( memberOut->getSubType() == hkClassMember::TYPE_CSTRING ) // array of c-strings
			{
				HK_ASSERT2(0x4765fb5e, 0, "Type not supported by hkRelArray");
			}
			else if( memberOut->getSubType() == hkClassMember::TYPE_STRINGPTR ) // array of c-string ptrs
			{
				HK_ASSERT2(0x4765fb5e, 0, "Type not supported by hkRelArray");
			}
			else if( memberOut->getSubType() == hkClassMember::TYPE_STRUCT ) // array of struct
			{
				const hkClass* sin = memberIn->getClass();
				const hkClass* sout = memberOut->getClass();
				if( sin && sout )
				{
					const char* cur = static_cast<const char*>(pointer);
					for( int i = 0; i < size; ++i )
					{
						saveBody(cur + i*sin->getObjectSize(), *sin, dataOut, *sout, zeroArray );
					}
					// Only POD structs are supported
// 						for( int j = 0; j < carray->size; ++j )
// 						{
// 							saveExtras(cur + j*sin->getObjectSize(), *sin,
// 								oa, *sout,
// 								structStart+j*sout->getObjectSize(), fixups, level);
// 						}
				}
			}
			else if( memberOut->getSubType() == hkClassMember::TYPE_VARIANT )
			{
				HK_ASSERT2(0x4765fb5e, 0, "Type not supported by hkRelArray");
			}
			else
			{
				if( memberOut->getSubType() == hkClassMember::TYPE_ULONG ) // array of Ulong type
				{
					writeUlongArray( dataOut, m_layoutOut.getRules().m_bytesInPointer, size, pointer );
				}
				else // array of POD type
				{
					writePodArray( dataOut, memberOut->getSubType(), memberOut->getArrayMemberSize(), size, pointer );
				}
			}
		}
	}
}

static inline hkResult saveCstring(const char* dataIn, int klassMemOffset, int extrasStart, hkOArchive& dataOut, hkRelocationInfo& fixups)
{
	objectCopierPadUp(dataOut.getStreamWriter(), 2);
	fixups.addLocal( klassMemOffset, extrasStart + dataOut.getStreamWriter()->tell() );
	dataOut.writeRaw( dataIn, hkString::strLen(dataIn)+1 );
	return HK_SUCCESS;
}

void hkObjectCopier::saveExtras(
	const void* dataIn,
	const hkClass& klassIn,
	hkOArchive& dataOut,
	const hkClass& klassOut,
	int classStart, // offset of where dataIn is stored
	int extrasStart,
	hkRelocationInfo& fixups, // fixups to apply on load
	hkLocalArray<char>& zeroArray,
	int level
)
{
	if ( level == 0 )
	{
		// assumes vtable is at start
		fixups.addFinish( classStart, klassOut.getName() );
	}
	++level;
	
	for( int memberIdx = 0, numMembers = klassOut.getNumMembers(); memberIdx < numMembers; ++memberIdx )
	{
		const hkClassMember& memberOut = klassOut.getMember(memberIdx);

		if( const hkClassMember* memberInPtr = (&klassIn==&klassOut) ? &memberOut : klassIn.getMemberByName( memberOut.getName() ) )
		{
			const hkClassMember& memberIn = *memberInPtr;
			const void* addressIn = static_cast<const char*>(dataIn) + memberIn.getOffset();

			if( !areMembersCompatible(memberIn, memberOut) )
			{
				//HK_WARN(0x337d3f11, "fixme: member '" << klassIn.getName() << "::" << memberIn.getName() << "' type has changed");
				continue;
			}
			if( m_flags.get(FLAG_RESPECT_SERIALIZE_IGNORED) && memberOut.getFlags().get(hkClassMember::SERIALIZE_IGNORED))
			{
				continue;
			}

			switch( memberOut.getType() )
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
				case hkClassMember::TYPE_FLAGS:
				case hkClassMember::TYPE_RELARRAY:
				{
					break; // nothing extra for these
				}
				case hkClassMember::TYPE_CSTRING:
				{
					const char* ptrTo = *static_cast<const char*const*>(addressIn);
					if( ptrTo != HK_NULL )
					{
						saveCstring(ptrTo, classStart + memberOut.getOffset(), extrasStart, dataOut, fixups);
					}
					break;
				}
				case hkClassMember::TYPE_STRINGPTR:
				{
					const hkStringPtr& ptrTo = *static_cast<const hkStringPtr*>(addressIn);
					if( ptrTo.cString() != HK_NULL )
					{
						saveCstring(ptrTo, classStart + memberOut.getOffset(), extrasStart, dataOut, fixups);
					}
					break;
				}
				case hkClassMember::TYPE_POINTER:
				{
					HK_ASSERT2(0x4ca4f06d,
						(memberOut.getSubType() != hkClassMember::TYPE_CSTRING
						&& memberOut.getSubType() != hkClassMember::TYPE_STRINGPTR
						&& memberOut.getSubType() != hkClassMember::TYPE_POINTER), "The pointer to c-string and general pointers are unsupported.");
					int nelem = min2( objectCopier_calcCArraySize(memberIn), objectCopier_calcCArraySize(memberOut) );
					int memberInSize = memberIn.getSizeInBytes()/objectCopier_calcCArraySize(memberIn);
					for( int i = 0; i < nelem; ++i )
					{
						const void* ptrFrom = static_cast<const char*>(addressIn) + i * memberInSize;
						const void* ptrTo = *static_cast<const void*const*>(ptrFrom);
						if( ptrTo != HK_NULL )
						{
							int memOffsetOut = memberOut.getOffset() + i * m_layoutOut.getRules().m_bytesInPointer;
							if( memberOut.getSubType() == hkClassMember::TYPE_CHAR )
							{
								saveCstring(static_cast<const char*>(ptrTo), classStart + memOffsetOut, extrasStart, dataOut, fixups);
							}
							else
							{
								if(memberOut.getSubType() != hkClassMember::TYPE_VOID)
								{
									fixups.addGlobal( classStart + memOffsetOut, const_cast<void*>(ptrTo), const_cast<hkClass*>(memberIn.getClass()) );
								}
							}
						}
					}
					break;
				}
				case hkClassMember::TYPE_FUNCTIONPOINTER:
				{
					//XXX add external
					break;
				}
				case hkClassMember::TYPE_ARRAY:
				case hkClassMember::TYPE_INPLACEARRAY:
				case hkClassMember::TYPE_SIMPLEARRAY:
				{
					const ObjectCopier_DummyArray<char>* carray = static_cast<const ObjectCopier_DummyArray<char>*>(addressIn);

					if( carray->size )
					{
						HK_ASSERT( 0xc585890e, carray->size >= 0 );
						if( memberOut.getSubType() == hkClassMember::TYPE_VOID )
						{
							// array of unknown types, skip
						}
						else if( memberOut.getSubType() == hkClassMember::TYPE_POINTER ) // array of pointer
						{
							fixups.addLocal( classStart + memberOut.getOffset(), extrasStart + dataOut.getStreamWriter()->tell() );
							hkUint64 zero = 0;
							HK_COMPILE_TIME_ASSERT( sizeof(void*) <= sizeof(zero) );
							int memberInElementSize = memberIn.getArrayMemberSize();
							const ObjectCopier_DummyArray<char>* rawarray = static_cast<const ObjectCopier_DummyArray<char>*>(addressIn);
							for( int i = 0; i < rawarray->size; ++i )
							{
								void* p = *reinterpret_cast<void**>(rawarray->data + i*memberInElementSize);
								if( p )
								{
									if(memberOut.getSubType() != hkClassMember::TYPE_VOID)
									{
										fixups.addGlobal(
											extrasStart + dataOut.getStreamWriter()->tell(),
											p,
											const_cast<hkClass*>(memberIn.getClass()) );
									}
								}
								dataOut.writeRaw( &zero, m_layoutOut.getRules().m_bytesInPointer );
							}
						}
						else if( memberOut.getSubType() == hkClassMember::TYPE_CSTRING ) // array of c-strings
						{
							fixups.addLocal( classStart + memberOut.getOffset(), extrasStart + dataOut.getStreamWriter()->tell() );

							hkUint64 zero = 0;
							HK_COMPILE_TIME_ASSERT( sizeof(char**) <= sizeof(zero) );
							const ObjectCopier_DummyArray<char*>* cstringArray = static_cast<const ObjectCopier_DummyArray<char*>*>(addressIn);
							
							int arrayStart = extrasStart + dataOut.getStreamWriter()->tell();
							for( int i = 0; i < cstringArray->size; ++i )
							{
								dataOut.writeRaw( &zero, m_layoutOut.getRules().m_bytesInPointer );
							}

							for( int i = 0; i < cstringArray->size; ++i)
							{
								if ( cstringArray->data[i] != HK_NULL )
								{
									PRINT(("\taddLocal at %p, cstring='%s'\n", arrayStart + i * m_layoutOut.getRules().m_bytesInPointer, cstringArray->data[i]));
									saveCstring(cstringArray->data[i], arrayStart + i * m_layoutOut.getRules().m_bytesInPointer, extrasStart, dataOut, fixups);
								}
							}
						}
						else if( memberOut.getSubType() == hkClassMember::TYPE_STRINGPTR ) // array of c-string ptrs
						{
							fixups.addLocal( classStart + memberOut.getOffset(), extrasStart + dataOut.getStreamWriter()->tell() );

							hkUint64 zero = 0;
							HK_COMPILE_TIME_ASSERT( sizeof(char**) <= sizeof(zero) );
							const ObjectCopier_DummyArray<hkStringPtr>* stringPtrArray = static_cast<const ObjectCopier_DummyArray<hkStringPtr>*>(addressIn);

							int arrayStart = extrasStart + dataOut.getStreamWriter()->tell();
							for( int i = 0; i < stringPtrArray->size; ++i )
							{
								dataOut.writeRaw( &zero, m_layoutOut.getRules().m_bytesInPointer );
							}

							for( int i = 0; i < stringPtrArray->size; ++i)
							{
								if ( stringPtrArray->data[i].cString() != HK_NULL )
								{
									PRINT(("\taddLocal at %p, cstring='%s'\n", arrayStart + i * m_layoutOut.getRules().m_bytesInPointer, static_cast<const char*>(stringPtrArray->data[i])));
									saveCstring(stringPtrArray->data[i], arrayStart + i * m_layoutOut.getRules().m_bytesInPointer, extrasStart, dataOut, fixups);
								}
							}
						}
						else if( memberOut.getSubType() == hkClassMember::TYPE_STRUCT ) // array of struct
						{
							int structStart = extrasStart + dataOut.getStreamWriter()->tell();
							fixups.addLocal( classStart + memberOut.getOffset(), structStart );
							const hkClass* sin = memberIn.getClass();
							const hkClass* sout = memberOut.getClass();
							if( sin && sout )
							{
								const char* cur = carray->data;
								for( int i = 0; i < carray->size; ++i )
								{
									saveBody(cur + i*sin->getObjectSize(), *sin, dataOut, *sout, zeroArray);
								}
								for( int j = 0; j < carray->size; ++j )
								{
									saveExtras(cur + j*sin->getObjectSize(), *sin,
										dataOut, *sout,
										structStart+j*sout->getObjectSize(), extrasStart, fixups, zeroArray, level);
								}
							}
						}
						else if( memberOut.getSubType() == hkClassMember::TYPE_VARIANT )
						{
							int arrayStart = extrasStart + dataOut.getStreamWriter()->tell();
							fixups.addLocal( classStart + memberOut.getOffset(), arrayStart );

							const ObjectCopier_DummyArray<hkVariant>* varray = static_cast<const ObjectCopier_DummyArray<hkVariant>*>(addressIn);

							for( int i = 0; i < varray->size; ++i )
							{
								hkUint64 aid[2] = {0,0}; // data, class pointer
								dataOut.writeArrayGeneric( aid, m_layoutOut.getRules().m_bytesInPointer, 2 );
							}
							for( int j = 0; j < varray->size; ++j )
							{
								int vstart = (2*j)*m_layoutOut.getRules().m_bytesInPointer;
								if( varray->data[j].m_object )
								{
									fixups.addGlobal(
										arrayStart + vstart,
										varray->data[j].m_object,
										varray->data[j].m_class );
								}
								if( varray->data[j].m_class )
								{
									fixups.addGlobal(
										arrayStart + vstart + m_layoutOut.getRules().m_bytesInPointer,
										const_cast<hkClass*>(varray->data[j].m_class),
										&hkClassClass );
								}
							}
						}
						else
						{
							fixups.addLocal( classStart + memberOut.getOffset(), extrasStart + dataOut.getStreamWriter()->tell() );
							if( memberOut.getSubType() == hkClassMember::TYPE_ULONG ) // array of Ulong type
							{
								writeUlongArray( dataOut,
												m_layoutOut.getRules().m_bytesInPointer, 
												carray->size, carray->data );
							}
							else // array of POD type
							{
								writePodArray(dataOut, memberOut.getSubType(), 
												memberOut.getArrayMemberSize(),
												carray->size, carray->data );
							}
						}
					}
					break;
				}
				case hkClassMember::TYPE_HOMOGENEOUSARRAY:
				{
					// class ptr, data ptr, size
					const ObjectCopier_DummyHomogeneousArray* darray = (const ObjectCopier_DummyHomogeneousArray*)addressIn;
					if( darray->klass )
					{
						const hkClass& sin = *(darray->klass);
						// Save the class pointer
						fixups.addGlobal(
							classStart + memberOut.getOffset(),
							darray->klass,
							const_cast<hkClass*>(&hkClassClass), true );

						int arrayStart = extrasStart + dataOut.getStreamWriter()->tell();

						// data pointer
						fixups.addLocal(
							classStart + memberOut.getOffset() + m_layoutOut.getRules().m_bytesInPointer,
							arrayStart );

						const char* cur = reinterpret_cast<const char*>( darray->data );
						const hkClass& sout = sin;
						HK_ASSERT( 0xc585890e, darray->size >= 0 );
						for( int i = 0; i < darray->size; ++i )
						{
							saveBody(cur + i*sin.getObjectSize(), sin, dataOut, sout, zeroArray );
						}
						for( int j = 0; j < darray->size; ++j )
						{
							saveExtras(cur + j*sin.getObjectSize(), sin,
								dataOut, sout,
								arrayStart+j*sout.getObjectSize(), extrasStart, fixups, zeroArray, level);
						}
					}
					else
					{
						HK_WARN(0x34f95a55, "No hkClass for " << klassIn.getName() << "::" << memberIn.getName() << "." );
					}
					break;
				}
				case hkClassMember::TYPE_STRUCT:
				{
					const hkClass* sin = memberIn.getClass();
					const hkClass* sout = memberOut.getClass();
					if( sin && sout )
					{
						int structStart = classStart + memberOut.getOffset();
						int nelemIn = objectCopier_calcCArraySize( memberIn );
						int nelemOut = objectCopier_calcCArraySize( memberOut );
						int nelem = min2(nelemIn, nelemOut);
						HK_ASSERT( 0xc585890e, nelem >= 0 );

						for( int i = 0 ; i < nelem; ++i )
						{
							const void* din = static_cast<const char*>(addressIn) + i * sin->getObjectSize();
							saveExtras( din, *sin, dataOut, *sout,
								structStart + i * sout->getObjectSize(), extrasStart, fixups, zeroArray, level);
						}
					}

					break;
				}
				case hkClassMember::TYPE_VARIANT:
				{
					int nelemIn = objectCopier_calcCArraySize( memberIn );
					int nelemOut = objectCopier_calcCArraySize( memberOut );
					int nelem = min2(nelemIn, nelemOut);

					for( int i = 0; i < nelem; ++i )
					{
						const void* vobject = static_cast<const char*>(addressIn) + (i*2  )*m_layoutIn.getRules().m_bytesInPointer;
						const void* vclass  = static_cast<const char*>(addressIn) + (i*2+1)*m_layoutIn.getRules().m_bytesInPointer;
						vobject = *static_cast<const void*const*>(vobject);
						vclass  = *static_cast<const void*const*>(vclass);
						if( vobject )
						{
							fixups.addGlobal(
								classStart + memberOut.getOffset() + (2*i  )*m_layoutOut.getRules().m_bytesInPointer,
								const_cast<void*>(vobject),
								static_cast<const hkClass*>(vclass) );
						}
						if( vclass )
						{
							fixups.addGlobal(
								classStart + memberOut.getOffset() + (2*i+1)*m_layoutOut.getRules().m_bytesInPointer,
								const_cast<void*>(vclass),
								&hkClassClass );
						}
					}

					break;
				}
				case hkClassMember::TYPE_ZERO:
				{
					HK_ERROR(0x641e3e05, "TYPE_ZERO should not occur.");
					break;
				}
				default:
				{
					HK_ERROR(0x641e3e05, "Unknown class member found during write of data.");
				}
			}
			objectCopierPadUp(dataOut.getStreamWriter());
		}
	}
}

hkResult hkObjectCopier::copyObject( const void* dataIn, const hkClass& klassIn,
		   hkStreamWriter* dataOut, const hkClass& klassOut, hkRelocationInfo& reloc )
{
	PRINT(("Starting %s %p at %x\n", klassIn.getName(), dataIn, dataOut->tell()));

#ifdef HK_DEBUG
	int objectStart = dataOut->tell();
	int origLocal = reloc.m_local.getSize();
	int origGlobal = reloc.m_global.getSize();
	int origVirtual = reloc.m_finish.getSize();
	int origImports = reloc.m_imports.getSize();
#endif

	hkLocalArray<char> save(1024);
	hkLocalArray<char> zero(16);

	hkArrayStreamWriter writer( &save, hkArrayStreamWriter::ARRAY_BORROW );
	hkOArchive oa( &writer, m_byteSwap );

	HK_ASSERT(0x9fa0fe6, (dataOut->tell() & (HK_REAL_ALIGNMENT-1))==0);

	// pass 1 - save the class data itself. hkRelArrays are also part of the body
	int classStart = dataOut->tell();
	saveBody( dataIn, klassIn, oa, klassOut, zero );
	dataOut->write( save.begin(), save.getSize() );
	objectCopierPadUp(dataOut);
	writer.clear();

	// pass 2 - save arrays etc
	int extrasStart = dataOut->tell();
	saveExtras( dataIn, klassIn, oa, klassOut, classStart, extrasStart, reloc, zero );
	dataOut->write( save.begin(), save.getSize() );
	objectCopierPadUp(dataOut);
	writer.clear();

#ifdef HK_DEBUG
	const char ERROR_STRING[] = "Fixup out of range in platform write.";
	{
		int i;
		int objectEnd = dataOut->tell();
		for( i = origLocal; i < reloc.m_local.getSize(); ++i )
		{
			hkRelocationInfo::Local& fix = reloc.m_local[i];
			HK_ASSERT2(0x4ea064f8, inRange(fix.m_fromOffset, objectStart, objectEnd), ERROR_STRING  );
			HK_ASSERT2(0x68ca252c, inRange(fix.m_toOffset, objectStart, objectEnd), ERROR_STRING );
		}
		for( i = origGlobal; i < reloc.m_global.getSize(); ++i )
		{
			hkRelocationInfo::Global& fix = reloc.m_global[i];
			HK_ASSERT2(0x747a20cd, inRange(fix.m_fromOffset, objectStart, objectEnd), ERROR_STRING );
		}
		for( i = origVirtual; i < reloc.m_finish.getSize(); ++i )
		{
			hkRelocationInfo::Finish& fix = reloc.m_finish[i];
			HK_ASSERT2(0x24c38635, inRange(fix.m_fromOffset, objectStart, objectEnd), ERROR_STRING );
		}
		for( i = origImports; i < reloc.m_imports.getSize(); ++i )
		{
			hkRelocationInfo::Import& fix = reloc.m_imports[i];
			HK_ASSERT2(0x3fe40756, inRange(fix.m_fromOffset, objectStart, objectEnd), ERROR_STRING );
		}
	}
#endif

	return oa.isOk() ? HK_SUCCESS : HK_FAILURE;
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
