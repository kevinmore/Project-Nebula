/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Tagfile/Binary/hkBinaryTagfileCommon.h>
#include <Common/Base/System/Io/OArchive/hkOArchive.h>
#include <Common/Base/System/Io/Writer/hkStreamWriter.h>
#include <Common/Serialize/Data/hkDataObject.h>
#include <Common/Serialize/Tagfile/Binary/hkBinaryTagfileWriter.h>
#include <Common/Serialize/Data/hkDataObjectImpl.h>
#include <Common/Base/Config/hkConfigVersion.h>

//todo: imports + exports

#if 0
#include <Common/Base/DebugUtil/Logger/hkOstreamLogger.h>
#define LOG(A) if(m_log) m_log->A
#else
class hkOstreamLogger;
#define LOG(A)
#endif


// The tagfile format typically contains in this order:
// * a header (TAG_FILE_INFO)
// * several classes (TAG_METADATA)
// * several objects (TAG_OBJECT_*)

// Differences between file versions:
//
// Version 0: dumb strings, recursive objects
// Version 1: string table, recursive objects
// Version 2:
//    optional dumb or table strings
//    flattened objects
//    don't use TAG_OBJECT_NULL, object id 0 is always the null object
// Version 3: store a hint about the size of int arrays
//    so loaded values aren't always 64-bit
//
// ---Object Writing Style---
//
// In the first "recursive" style, the object members were saved recursively
// and in depth first order. Only objects which were referenced more than
// once had a TAG_OBJECT_REMEMBER so it could be referenced
// (TAG_OBJECT_BACKREF) later. Objects referenced once don't get a label.
// TAG_OBJECT_REMEMBER_AUTO is not used in this style.
// 
// The format was then changed to "flattened" style. The writing order changed
// to be more similar to the old packfile.
// That is, every time a pointer is encountered we save an id and thus the
// format is much more flat and uses less recursion when reading & writing.
// TAG_OBJECT_REMEMBER and TAG_OBJECT are not used in this style.
//
// ---String Handling Style---
// 
// "Dumb" string handling outputs every string used, using back references
// to eliminate subsequent references. The backref indices are unique for strings
// and don't overlap with object backrefs.
//
// "Table" strings uses a preloaded table (HK_TAG_STRING_LIST) for common strings.
// This table is pre-loaded string list so that these strings are not emitted to
// the output file, just referenced using backrefs as before.

namespace
{
	struct Writer
	{
		struct HandleOps
		{
			inline static unsigned hash( const hkDataObject::Handle& key, unsigned mod )
			{
				return unsigned((hkUlong(key.p0) >> 4) * 2654435761U) & mod;
			}
			inline static void invalidate( hkDataObject::Handle& key )
			{
				key.p0 = (void*)hkUlong(-1);
			}
			inline static hkBool32 isValid( hkDataObject::Handle& key )
			{
				return key.p0 != (void*)hkUlong(-1);
			}
			inline static hkBool32 equal( const hkDataObject::Handle& key0, const hkDataObject::Handle& key1 )
			{
				return key0.p0 == key1.p0 && key0.p1 == key1.p1;
			}
		};
		typedef hkDataObjectImpl ObjectImpl;
		typedef hkDataClassImpl ClassImpl;
		typedef hkMap<hkDataObject::Handle, int, HandleOps> DoneObjects;
		typedef hkStringMap<int> DoneClasses;

		struct ObjectDetails
		{
			hkDataObject::Handle handle;
			int bitfieldStartIndex; // Bitfield buffer is shared - remember where I begin, we know the size from the object
			int objectRememberTag; // -1 for structs, valid id for objects
		};

		Writer( hkStreamWriter* sw, hkTagfileWriter::AddDataObjectListener* listener, const hkTagfileWriter::Options& options )
			: m_oa(sw)
			, m_currentRememberTag(1) // 0 is null object
			, m_listenerCache(listener)
			, m_log(HK_NULL)
			, m_writeSerializeIgnored(options.m_writeSerializeIgnored)
		{
			m_doneClasses.insert("",0);
			m_writtenStrings.insert("", 0);
			hkDataObject::Handle nullObject = {0,0};
			m_detailIndexFromObject.insert( nullObject, 0 );
			m_bitfieldBuf.reserve(512);
			m_details.reserve(128);
			ObjectDetails& nullDetails = this->m_details.expandOne();
			nullDetails.bitfieldStartIndex = -1;
			nullDetails.handle = nullObject;
			nullDetails.objectRememberTag = 0;
			//m_log = new hkOstreamLogger("tlogw.txt");
		}

		~Writer()
		{
			//delete m_log;
		}

		void writeInt(hkInt64 i)
		{
			// Write a single integer

			// We use a variable length encoding. The top bit of each byte indicates continuation bytes.
			// The lower bits of each byte are then combined into an integer.
			// e.g. [0]011 0100 -> (bottom 7 bits) -> 011 0100 -> 0x34
			// e.g. [1]000 1111  [0]001 0100 -> 111 1001 0100 -> 0x794

			// Using this encoding, negative numbers would be huge because of 2s complement.
			// Therefore we use the lowest bit as a sign bit.
			// e.g. input:output 0:0 -1:11 1:10 -2:101 2:100 ...
			hkUint64 u;
			if(hkUint64(i) == 0x8000000000000000ll)
			{
				// This is a special case as -i will overflow to 0
				m_oa.write8u(0x81);
				u = 0x0200000000000000ll;
			}
			else if( i < 0 )
			{
				u = ((-i) << 1) | 1;
			}
			else
			{
				u = i << 1;
			}
			while( true )
			{
				const int mask = 0x7f;
				hkUint8 out = hkUint8(mask & u);
				u >>= 7;
				if( u )
				{
					m_oa.write8u( out | 0x80 );
				}
				else
				{
					m_oa.write8u( out );
					break;
				}
			}
		}

			// todo ; only memoized some strings (e.g. mem names etc)
		void writeString(const char* s)
		{
			// String is written as integer len followed by char[len] bytes
			// Negative len is a special case. -1 means null string (as opposed to empty)
			// Other negative numbers are used as backrefs to previously written strings
			if( s )
			{
				int prevId = m_writtenStrings.getWithDefault(s, -1);
				if( prevId == -1 )
				{
					int len = hkString::strLen(s);
					writeInt( len );
					m_oa.writeRaw(s,len);
					m_writtenStrings.insert(s, m_writtenStrings.getSize()+1); // index starts at -2. 0 and -1 are reserved for ["", HK_NULL]
				}
				else
				{
					writeInt( -prevId );
				}
			}
			else
			{
				writeInt(-1); // null
			}
		}
		void writeFloat(hkFloat32 r)
		{
			m_oa.writeFloat32(r);
		}

		void writeFloatArray(const hkFloat32* r, int n)
		{
			m_oa.writeArrayFloat32( r, n );
		}

		void writeFloatArray(const hkDouble64* r, int n)
		{
			m_oa.writeArrayFloat32( r, n );
		}

		int getWrittenClassId( const hkDataClass& klass )
		{
			if( klass.getImplementation() == HK_NULL )
			{
				return 0;
			}

			int clsId = m_doneClasses.getWithDefault( klass.getName(), -1 );
			HK_ASSERT(0x70a8ef40, clsId > 0);
			return clsId;
		}

		int writeClass( const hkDataClass& klass )
		{
			// todo write signature only.
			if( klass.getImplementation() == HK_NULL )
			{
				return 0;
			}

			int clsId = m_doneClasses.getWithDefault( klass.getName(), -1 );
			if( clsId >= 0 )
			{
				return clsId;
			}
			// write parent class
			int parId = writeClass( klass.getParent() );

			// need to check again: writing parent may have written self.
			// e.g. struct A { B* b; }; struct B : public A {};
			clsId = m_doneClasses.getWithDefault( klass.getName(), -1 );
			if( clsId == -1 )
			{
				// assign an index
				clsId = m_doneClasses.getSize();
				m_doneClasses.insert( klass.getName(), clsId );

				LOG(debug("class %s %i members", klass.getName(), klass.getNumDeclaredMembers()));

				// write this class
				int numMembers = klass.getNumDeclaredMembers();
				hkArray<hkDataClass::MemberInfo>::Temp minfos(numMembers);
				klass.getAllDeclaredMemberInfo( minfos );
				writeInt(hkBinaryTagfile::TAG_METADATA);
				writeString( klass.getName() );
				writeInt( klass.getVersion() );
				if( klass.getVersion() < 0 )
				{
					HK_WARN(0x6c482a3d, "Serializing class " << klass.getName() << " which is marked as under development (negative version number). "
						"No patching will be done when loading which means it may load incorrectly in the future.");
				}
				writeInt( parId );
				writeInt( numMembers );
				for( int memberIndex = 0; memberIndex < numMembers; ++memberIndex )
				{
					const hkDataClass::MemberInfo& mi = minfos[memberIndex];
					writeString( mi.m_name );

					const char* className;
					int tupleCount;
					hkLegacyType::Type type = hkTypeManager::getLegacyType(mi.m_type, &className, tupleCount);

					writeInt( type );
					// possibly tuple size
					if( type & hkLegacyType::TYPE_TUPLE )
					{
						HK_ASSERT(0x6449be65, tupleCount);
						writeInt( tupleCount );
					}
					else
					{
						HK_ASSERT(0x3fec8c18, tupleCount == 0);
					}
					// class type
					int bt = type & hkLegacyType::TYPE_MASK_BASIC_TYPES;
					if( bt == hkLegacyType::TYPE_STRUCT || bt == hkLegacyType::TYPE_OBJECT )
					{
						if( className ) // variants have null class
						{
							// Need to look up the class to get the name, as writeString relies on 
							// the strings be memorized ones
							hkDataClassImpl* clsImpl = klass.getWorld()->findClass(className);
							HK_ASSERT(0x32432432, clsImpl);

							hkDataClass cls(clsImpl);
							writeString( cls.getName());
						}
						else
						{
							writeString(HK_NULL);
						}
					}
					else
					{
						HK_ASSERT(0x4a743631, className == HK_NULL );
					}
				}

				// write classes referenced by members. Do this after the main body
				// to avoid circular references e.g. struct Foo { Foo* m_next; }
				for( int memberIndex = 0; memberIndex < numMembers; ++memberIndex )
				{
					hkDataObject::Type term = minfos[memberIndex].m_type->findTerminal();

					if( term->isClass())
					{
						hkDataClassImpl* clsImpl = klass.getWorld()->findClass(term->getTypeName());
						HK_ASSERT(0x32432432, clsImpl);

						hkDataClass cls(clsImpl);

						writeClass(cls);
					}
				}
			}
			return clsId;
		}

		template<typename VALUE>
		hkBool32 worthWriting( const VALUE& value )
		{
			hkDataObject::Type type = value.getType();

			switch( type->getSubType())
			{
				case hkTypeManager::SUB_TYPE_BYTE:
				case hkTypeManager::SUB_TYPE_INT:
				{
					return value.asInt64() != 0;
				}
				case hkTypeManager::SUB_TYPE_REAL:
				{
					return value.asReal() != 0;
				}
				case hkTypeManager::SUB_TYPE_TUPLE:
				{
					return true;
				}
				case hkTypeManager::SUB_TYPE_POINTER:
				case hkTypeManager::SUB_TYPE_CLASS:
				{
					return value.asObject().isNull() == hkFalse32;
				}
				case hkTypeManager::SUB_TYPE_ARRAY:
				{
					return value.asArray().getSize() != 0;
				}
				case hkTypeManager::SUB_TYPE_CSTRING:
				{
					const char* s = value.asString();
					return s != 0;
				}
				case hkTypeManager::SUB_TYPE_VOID: // for +nosave
				{
					return false;
				}
				default:
				{
					HK_ASSERT(0x43e4b8f8, 0);
					return false;
				}
			}
		}

		struct MemberWriteFlags
		{
			MemberWriteFlags( int nm )
			{
				m_flags.setSize( HK_NEXT_MULTIPLE_OF(8,nm), 0 );
				m_flags.setSize(nm);
			}
			int getNumBytes() const
			{
				return HK_NEXT_MULTIPLE_OF(8,m_flags.getSize()) / 8;
			}
			void fromBytes( const hkUint8* inb, int numBytes )
			{
				int origSize = m_flags.getSize();  // pad up & restore to make logic easier
				m_flags.setSize( HK_NEXT_MULTIPLE_OF(8,origSize), 0 );
				HK_ASSERT(0x157006c9, m_flags.getSize()/8 == numBytes);

				int flagIdx = 0;
				for( int byteIndex = 0; byteIndex < numBytes; byteIndex += 1 )
				{
					hkUint8 cur = inb[byteIndex];
					for( int bit = 0; bit < 8; bit += 1 )
					{
						m_flags[flagIdx++] = cur & 1;
						cur >>= 1;
					}
				}

				m_flags.setSize(origSize);
				HK_ASSERT(0x4500fd83, isOk());
			}
			void toBytes( hkUint8* out )
			{
				HK_ASSERT(0x4500fd82, isOk());
				int origSize = m_flags.getSize(); // pad up & restore to make logic easier
				m_flags.setSize( HK_NEXT_MULTIPLE_OF(8,origSize), 0 );

				for( int i = 0; i < m_flags.getSize(); i += 8 )
				{
					*out++ = m_flags[i+0] << 0
						| m_flags[i+1] << 1
						| m_flags[i+2] << 2
						| m_flags[i+3] << 3
						| m_flags[i+4] << 4
						| m_flags[i+5] << 5
						| m_flags[i+6] << 6
						| m_flags[i+7] << 7;
				}

				m_flags.setSize(origSize);
			}
			hkUint8& memberInteresting( int idx )
			{
				HK_ASSERT(0x7216abf7, (m_flags[idx] & ~1) == 0 );
				return m_flags[idx];
			}
			hkBool32 isOk() const
			{
				for( int i = 0; i < m_flags.getSize(); ++i )
				{
					if( m_flags[i] & ~1 )
					{
						return false;
					}
				}
				return true;
			}

			hkArray<hkUint8>::Temp m_flags;
		};

		struct TodoItem
		{
			hkDataObject::Handle handle;
			hkBool32 isStruct;
			void set(const hkDataObject::Handle& h, hkBool32 is)
			{
				handle = h; 
				isStruct = is;
			}
		};

		inline hkBool32 mayContainPointers(hkDataObject::Type t)
		{
			return t->findTerminal()->isClass();
		}

		void scanArray(hkDataArray& arrToScan, hkArray<TodoItem>::Temp& objsTodo, hkBool32 immediate=false)
		{
			hkDataObject::Type type = arrToScan.getType();

			switch (type->getSubType())
			{
				case hkTypeManager::SUB_TYPE_POINTER:
				{
					// Object
					int asize = arrToScan.getSize();
					for( int i = 0; i < asize; ++i )
					{
						objsTodo.expandOne().set( arrToScan[i].asObject().getHandle(), false );
					}
					break;
				}
				case hkTypeManager::SUB_TYPE_CLASS:
				{
					// Struct
					hkDataClass k = arrToScan.getClass();
					writeClass(k);
					for( int memberIndex = 0; memberIndex < k.getNumMembers(); ++memberIndex )
					{
						hkDataClass::MemberInfo minfo;
						k.getMemberInfo(memberIndex, minfo);

						hkDataObject::Type memType = minfo.m_type;
						switch (memType->getSubType())
						{
							case hkTypeManager::SUB_TYPE_POINTER:
							{
								hkDataArray objs = arrToScan.swizzleObjectMember( minfo.m_name );
								int asize = objs.getSize();
								for( int i = 0; i < asize; ++i )
								{
									objsTodo.expandOne().set( objs[i].asObject().getHandle(), false );
								}
								break;
							}
							case hkTypeManager::SUB_TYPE_CLASS:
							{
								hkDataArray objs = arrToScan.swizzleObjectMember( minfo.m_name );
								int asize = objs.getSize();
								//HK_ASSERT2(0x22f57705, asize<10, "performance warning");
								for( int i = 0; i < asize; ++i )
								{
									scanObjectForPointers( objs[i].asObject(), objsTodo, true );
								}
								break;
							}
							case hkTypeManager::SUB_TYPE_ARRAY:
							{
								hkDataArray childArray = arrToScan.swizzleObjectMember( minfo.m_name );
								scanArray(childArray, objsTodo, immediate);
								/*
								const int size = arrToScan.getSize();
								for (int i = 0; i < size; i++)
								{
									hkDataArray childArray = arrToScan[i].asArray();
									scanArray(childArray, objsTodo, immediate);
								} */
								break;
							}
							case hkTypeManager::SUB_TYPE_TUPLE:
							default: 
							{ 
								break;
							}
						}
					}
					break;
				}
				case hkTypeManager::SUB_TYPE_TUPLE:
				case hkTypeManager::SUB_TYPE_ARRAY:
				{
					const int size = arrToScan.getSize();
					for (int i = 0; i < size; i++)
					{
						hkDataArray arr = arrToScan[i].asArray();
						scanArray(arr, objsTodo, immediate);
					}
					break;
				}
				default: 
				{
					break;
				}
			}
		}

		void scanObjectForPointers(const hkDataObject& objToScan, hkArray<TodoItem>::Temp& objsTodo, hkBool32 immediate=false)
		{
			hkDataClass klass = objToScan.getClass();
			for(hkDataObject::Iterator memIt = objToScan.getMemberIterator();
				objToScan.isValid(memIt);
				memIt = objToScan.getNextMember(memIt) )
			{
				const hkDataObject::Value value = objToScan.getMemberValue(memIt);
				hkDataObject::Type type = value.getType();

					// only interested in scanning types containing other objects
				if( mayContainPointers(type) )
				{
					switch( type->getSubType() )
					{
						case hkTypeManager::SUB_TYPE_POINTER:
						{
							objsTodo.expandOne().set( value.asObject().getHandle(), false );
							break;
						}
						case hkTypeManager::SUB_TYPE_CLASS:
						{
							if( !immediate ) // the normal case
							{
								objsTodo.expandOne().set( value.asObject().getHandle(), true );
							}
							else // we have been called from  TYPE_(ARRAY|TUPLE)_STRUCT, it must be structs all the way down
							{
								scanObjectForPointers( value.asObject(), objsTodo, true );
							}
							break;
						}
						case hkTypeManager::SUB_TYPE_TUPLE:
						case hkTypeManager::SUB_TYPE_ARRAY:
						{
							hkDataArray childArray = value.asArray();
							scanArray(childArray, objsTodo, immediate);
							break;
						}
						default:
						{
							HK_ASSERT2(0x4e628c85, 0, "Internal error, missing case statement");
							break;
						}
					}
				}
			}
		}

			// Recursively scan objects for addition to packfile
		void scanObjectTree( const hkDataWorld* world, const hkDataObject& topObj, hkBool32 topIsStruct )
		{
			hkArray<TodoItem>::Temp objsTodo; objsTodo.reserve(128);
			objsTodo.expandOne().set(topObj.getHandle(), topIsStruct);
			
				// For each object...
			while( objsTodo.getSize() )
			{
				TodoItem curTodo = objsTodo.back();
				objsTodo.popBack();
				if( curTodo.handle.p0 == HK_NULL && curTodo.handle.p1 == HK_NULL )
				{
					continue;
				}
				hkDataObject objSupplied = world->findObject( curTodo.handle );
				const hkDataObject objToScan = m_listenerCache.callListenerAndCache(objSupplied);

				if( objToScan.isNull() )
				{
					continue;
				}
				if( m_detailIndexFromObject.getWithDefault( objToScan.getHandle(), -1 ) != -1 )
				{
					continue; // we've seen it before, next
				}

					// save metadata now so classes all appear first
				hkDataClass klass = objToScan.getClass();
				writeClass( klass );

					// remember also which members are worth writing (non-default values)
				MemberWriteFlags memberFlags( klass.getNumMembers() );
				for(hkDataObject::Iterator memIt = objToScan.getMemberIterator();
					objToScan.isValid(memIt);
					memIt = objToScan.getNextMember(memIt) )
				{
					const hkDataObject::Value value = objToScan.getMemberValue(memIt);
					if( !worthWriting(value) )
					{
						continue;
					}
					const char* memName = objToScan.getMemberName(memIt);
					int memIndex = klass.getMemberIndexByName(memName);
					memberFlags.memberInteresting(memIndex) = true;
				}
					// convert bit array to bitfield and store
				{
					int objectDetailsIndex = m_detailIndexFromObject.getSize();
					m_detailIndexFromObject.insert( objToScan.getHandle(), objectDetailsIndex );
					ObjectDetails& details = m_details.expandOne();
					details.handle = objToScan.getHandle();
					details.bitfieldStartIndex = m_bitfieldBuf.getSize();
					details.objectRememberTag = curTodo.isStruct ? -1 : m_currentRememberTag++;
					hkUint8* bf = m_bitfieldBuf.expandBy( memberFlags.getNumBytes() ); // already rounded up to multiple of 8
					memberFlags.toBytes( bf ); 			
					HK_ASSERT(0x1a33ddaf, m_details.getSize() == m_detailIndexFromObject.getSize() );
				}

				// Look for more pointers in our members
				scanObjectForPointers(objToScan, objsTodo);

			} // while(objsTodo)
		}

		void _writeArrayItems( const hkDataArray& arr, int asize )
		{
			hkDataObject::Type type = arr.getType();
			if( type->isTuple() && !type->isVec())
			{
				for( int i = 0; i < asize; ++i )
				{
					hkDataArray a = arr[i].asArray();
					_writeArrayItems( a, a.getSize() );
				}
				return;
			}
			switch( type->getSubType())
			{
				case hkTypeManager::SUB_TYPE_VOID:
				{
					return; //TODO
				}
				case hkTypeManager::SUB_TYPE_BYTE:
				{
					for( int i = 0; i < asize; ++i )
					{
						m_oa.write8u( hkUint8(arr[i].asInt()) );
					}
					break;
				}
				case hkTypeManager::SUB_TYPE_INT:
				{
					if(arr.getUnderlyingIntegerSize() == hkSizeOf(hkInt32))
					{
						writeInt(hkSizeOf(hkInt32));
						for( int i = 0; i < asize; ++i )
						{
							writeInt( arr[i].asInt() );
						}
					}
					else
					{
						writeInt(hkSizeOf(hkInt64));
						for( int i = 0; i < asize; ++i )
						{
							writeInt( arr[i].asInt64() );
						}
					}

					break;
				}
				case hkTypeManager::SUB_TYPE_REAL:
				{
					for( int i = 0; i < asize; ++i )
					{
						writeFloat( hkFloat32(arr[i].asReal()) );
					}
					break;
				}
				case hkTypeManager::SUB_TYPE_TUPLE:
				{
					if (type->isVec())
					{
						const int numReal = type->getTupleSize();
						int numRealOut = numReal;

						// Special case 4, if the last is zero, don't write it
						if (numReal == 4)
						{
							numRealOut = 3;
							const hkReal realZero = 0.0f;
							for( int i = 0; i < asize; ++i )
							{
								const hkReal* r = arr[i].asVec(numReal);
								if ( hkString::memCmp(&r[3], &realZero, sizeof(hkReal)) != 0 )
								{
									numRealOut = 4;
									break;
								}
							}
							
							writeInt( numRealOut );
						}

						for( int i = 0; i < asize; ++i )
						{
							const hkReal* r = arr[i].asVec(numReal);
							writeFloatArray( r, numRealOut );
						}
					}
					else
					{
						HK_ASSERT(0x3454535a, !"Unsupported array tuple type");
					}
					break;
				}
				case hkTypeManager::SUB_TYPE_CLASS:
				{
					int numMembers = arr.getClass().getNumMembers();
					hkArray<hkDataClass::MemberInfo>::Temp minfos(numMembers);
					arr.getClass().getAllMemberInfo( minfos );

					MemberWriteFlags flags( numMembers );
					{
						for( int memberIndex = 0; memberIndex < numMembers; ++memberIndex )
						{
							hkDataClass::MemberInfo& minfo = minfos[memberIndex];
							if( minfo.m_type->isVoid())
							{
								continue;
							}
							hkDataArray m = arr.swizzleObjectMember(minfo.m_name);
							for( int i = 0; i < m.getSize(); ++i )
							{
								if( worthWriting(m[i]) )
								{
									flags.memberInteresting(memberIndex) = true;
									break;
								}
							}
						}
						hkArray<hkUint8>::Temp buf;
						buf.setSize(flags.getNumBytes());
						flags.toBytes( buf.begin() );
						m_oa.writeRaw( buf.begin(), buf.getSize() );
					}
					for( int memberIndex = 0; memberIndex < numMembers; ++memberIndex )
					{
						if( flags.memberInteresting(memberIndex) )
						{
							hkDataClass::MemberInfo& minfo = minfos[memberIndex];
							hkDataArray m = arr.swizzleObjectMember(minfo.m_name);
							LOG(pushScope(minfo.m_name));
							_writeArrayItems(m, asize);
							LOG(popScope());
						}
					}
					break;
				}
				case hkTypeManager::SUB_TYPE_POINTER:
				{
					for( int i = 0; i < asize; ++i )
					{
						writeObjectReference( arr[i].asObject() );
					}
					break;
				}
				case hkTypeManager::SUB_TYPE_CSTRING:
				{
					for( int i = 0; i < asize; ++i )
					{
						const char* s = arr[i].asString();
						LOG(debug("String \"%s\"", s));
						writeString( s );
					}
					break;
				}
				case hkTypeManager::SUB_TYPE_ARRAY:
				{
					for ( int i = 0; i < asize; i++)
					{
						hkDataArray childArr = arr[i].asArray();

						const int childSize = childArr.getSize();
						// Write the array size
						writeInt(childSize);

						_writeArrayItems(childArr, childSize);
					}
					break;
				}
				default:
				{
					HK_ASSERT2(0x4e628c86, 0, "Unsupported array type");
					break;
				}
			}
		}
		
		void writeBinaryValue( const hkDataObject::Value& value, const hkDataClass::MemberInfo& minfo )
		{
			hkDataObject::Type type = value.getType();

			if (type->isTuple() && !type->isVec())
			{
				hkDataArray a = value.asArray();
				LOG(debug("Tuple of count %i, type %s", a.getSize(), type->asString()));
				_writeArrayItems( a, a.getSize() );
				return;
			}
			else if( type->isArray())
			{
				hkDataArray a = value.asArray();
				int asize = a.getSize();
				LOG(debug("Array of size %i, type %s", asize, type->asString()));
				writeInt( asize );

				HK_ASSERT(0x234324, minfo.m_type->isArray());

				hkDataObject::Type memParentType = minfo.m_type->getParent();
				if( memParentType->isClass() && memParentType->getTypeName() == HK_NULL )
				{
					// homogeneous array needs class id as it is not in the minfo
					hkDataObject::Type arrParentType = type->getParent();
					HK_ASSERT(0x453434aa, arrParentType->isClass());
					
					const hkDataWorld* world = value.m_impl->getClass()->getWorld();

					hkDataClass k = world->findClass(arrParentType->getTypeName());
					HK_ASSERT(0x5345abbb, k.getImplementation());

					int clsId = m_doneClasses.getWithDefault( k.getName(), 0 );
					HK_ASSERT(0x18d0df77, clsId > 0);
					writeInt(clsId);
				}

				_writeArrayItems( a, asize );
			}
			else
			{
				switch( type->getSubType())
				{
					case hkTypeManager::SUB_TYPE_BYTE:
					{
						unsigned v = unsigned(value.asInt());
						LOG(debug("Byte %x", v));
						m_oa.write8u( hkUint8(v) );
						break;
					}
					case hkTypeManager::SUB_TYPE_INT:
					{
						hkInt64 v = value.asInt64();
						LOG(debug("Int %x", v));
						writeInt( v );
						break;
					}
					case hkTypeManager::SUB_TYPE_REAL:
					{
						hkReal r = value.asReal();
						LOG(debug("Real %f", r));
						writeFloat(hkFloat32(r));
						break;
					}
					case hkTypeManager::SUB_TYPE_TUPLE:
					{
						if (type->isVec())
						{
							const int numReal = type->getTupleSize();
							LOG(debug("Vector%i", numReal));
							writeFloatArray( value.asVec(numReal), numReal );
							break;
						}
						HK_ASSERT(0x234a2432, !"Unhandled tuple type");
						break;
					}
					case hkTypeManager::SUB_TYPE_POINTER:
					{
						writeObjectReference( value.asObject() );
						break;
					}
					case hkTypeManager::SUB_TYPE_CLASS:
					{
						writeStructReal( value.asObject() );
						break;
					}
					case hkTypeManager::SUB_TYPE_CSTRING:
					{
						LOG(debug("String \"%s\"", value.asString()));
						writeString(value.asString());
						break;
					}
					default:
					{
						HK_ASSERT(0x4e628c85, 0);
						break;
					}
				}
			}
		}

			// Write an object index
		void writeObjectReference( const hkDataObject& obj )
		{
			const hkDataObject objToSave = m_listenerCache.getCachedDataObject(obj);
			int detailIndex = m_detailIndexFromObject.getWithDefault( objToSave.getHandle(), -1 ); 
			HK_ASSERT3(0x5bd3ee80, detailIndex != -1, "Expect object " << objToSave.getClass().getName() << ", handle = (" << objToSave.getHandle().p0 << " | " << objToSave.getHandle().p1 << ")." );
			int objectRememberTag = m_details[detailIndex].objectRememberTag;
			LOG(debug("Object *%08i", objectRememberTag));
			writeInt( objectRememberTag );
		}

			// actually write the object data
		void writeObjectReal( const hkDataObject& obj )
		{
			const hkDataObject objToSave = m_listenerCache.getCachedDataObject(obj);
			HK_ASSERT2(0x2129a89d, objToSave.isNull() == hkFalse32, "");
			ObjectDetails& details = m_details[ m_detailIndexFromObject.getWithDefault( objToSave.getHandle(), -1 ) ];
			HK_ASSERT2(0x7ed2236c, details.objectRememberTag != -1, "struct found where object expected");

			writeInt( hkBinaryTagfile::TAG_OBJECT_REMEMBER );
			hkDataClass klass = objToSave.getClass();
			writeInt( getWrittenClassId(klass) );
			_writeMembersCommon(obj, details); // objects are just structs with a little extra header stuff
		}

		void writeStructReal( const hkDataObject& obj )
		{
			HK_ASSERT2(0x2129a89e, obj.isNull() == hkFalse32, "shouldn't happen");
			HK_ASSERT2(0x2129a89f, m_listenerCache.getCachedDataObject(obj)==obj, "can't replace a struct" );

			int detailIndex = m_detailIndexFromObject.getWithDefault( obj.getHandle(), -1 ); 
			ObjectDetails& details = m_details[detailIndex];
			HK_ASSERT2(0x2582dbc4, details.objectRememberTag==-1, "object found where struct expected");
			LOG(pushScope("Members"));
			_writeMembersCommon(obj, details);
			LOG(popScope());
		}

		void _writeMembersCommon(const hkDataObject& obj, const ObjectDetails& details)
		{
			// bitfield present members
			hkDataClass klass = obj.getClass();
			int numKlassMembers = klass.getNumMembers();
			MemberWriteFlags flags( numKlassMembers );
			{
				int numBytes = flags.getNumBytes();
				hkArray<hkUint8>::Temp bitfield( &m_bitfieldBuf[details.bitfieldStartIndex], numBytes, numBytes );
				m_oa.writeRaw( bitfield.begin(), bitfield.getSize() );
				flags.fromBytes( bitfield.begin(), bitfield.getSize() );
			}

			// write values based on bitfield
			hkArray<hkDataClass::MemberInfo>::Temp memberInfos(numKlassMembers);
			klass.getAllMemberInfo(memberInfos);
			for( int memberIndex = 0; memberIndex < numKlassMembers; ++memberIndex )
			{
				const hkDataClass::MemberInfo& mi = memberInfos[memberIndex];
				if( flags.memberInteresting(memberIndex)  )
				{
					LOG(pushScope(mi.m_name));
					const hkDataObject::Value value = obj[ mi.m_name ];
					writeBinaryValue( value, mi );
					LOG(popScope());
				}
				else
				{
					LOG(debug("Skip %s", mi.m_name));
				}
			}
		}

		void writeHeader()
		{
			const int TAGFILE_VERSION = 4;
			m_oa.write32( hkBinaryTagfile::BINARY_MAGIC_0 );
			m_oa.write32( hkBinaryTagfile::BINARY_MAGIC_1 );
			writeInt( hkBinaryTagfile::TAG_FILE_INFO );
			writeInt( TAGFILE_VERSION );
			writeString( HAVOK_SDK_VERSION_STRING );
			LOG(debug("header version %i", TAGFILE_VERSION));

// 			int outputVersion
// 			if(outputVersion == 0)
// 			{
// 				// No extra initialization required
// 			}
// 			else if(outputVersion == 1)
// 			{
// 				writeInt(HK_TAG_LIST_VERSION);
// 				m_writtenStrings.clear();
// 				m_writtenStrings.insert("", 0);
// 				for(int i = 0; i < HK_NUM_TAG_STRINGS; i++)
// 				{
// 					m_writtenStrings.insert(HK_TAG_STRING_LIST[i], i + 2);
// 				}
// 			}
		}


			/// Two passes
			/// Pass1: Traverse tree, write metadata as we go
			///		For each object, remember bitfield of members to write out
			/// Pass2: Traverse tree, writing objects
		hkResult writeObjectTree(const hkDataObject& topObj)
		{
			const hkDataWorld* world = topObj.getClass().getWorld();

			writeHeader();
			scanObjectTree(world, topObj, false);
			HK_ON_DEBUG( int curObjectIndex = 1);
			for( int i = 1; i < m_details.getSize(); ++i )
			{
				ObjectDetails& details = m_details[i];
				if( details.objectRememberTag != -1 )
				{
					HK_ASSERT2(0x6323eed6, curObjectIndex++ == details.objectRememberTag,"");
					hkDataObject obj = world->findObject( details.handle );

					LOG(debug("Obj %s Remembered as &%08i", obj.getClass().getName(), details.objectRememberTag));
					LOG(pushScope("Members"));
					writeObjectReal(obj);
					LOG(popScope());
				}
			}
			writeInt(hkBinaryTagfile::TAG_FILE_END);
			return HK_SUCCESS;
		}

		class ListenerCache
		{
			public:

				ListenerCache(hkTagfileWriter::AddDataObjectListener* listener) : m_listener(listener)
				{
				}

				~ListenerCache()
				{
					for( int i = 0; i < m_cachedObjects.getSize(); ++i )
					{
						delete m_cachedObjects[i];
					}
				}

				hkDataObject getCachedDataObject(const hkDataObject& object)
				{
					if( m_listener && !object.isNull() )
					{
						hkDataObject::Handle h = object.getHandle();
						int index = m_indexFromHandle.getWithDefault(h, -1);
						if( index != -1 )
						{
							return m_cachedObjects[index]->getImplementation();
						}
					}
					return object;
				}

				hkDataObject callListenerAndCache(const hkDataObject& object)
				{
					if( m_listener && !object.isNull() )
					{
						hkDataObject::Handle h = object.getHandle();
						HK_ASSERT(0x36ed57e1, h.p0);
						int index = m_indexFromHandle.getWithDefault(h, -1);
						if( index == -1 )
						{
							hkDataObject newObject = m_listener->addDataObjectCallback(object);
							if( newObject != object )
							{
								m_indexFromHandle.insert(h, m_cachedObjects.getSize());
								m_cachedObjects.expandOne() = new hkDataObject(newObject);
								return newObject;
							}
						}
						else
						{
							return m_cachedObjects[index]->getImplementation();
						}
					}
					return object;
				}

			private:

				hkTagfileWriter::AddDataObjectListener* m_listener;
				DoneObjects m_indexFromHandle;
				hkArray<hkDataObject*>::Temp m_cachedObjects;
		};

		hkOArchive m_oa;
		hkStringMap<int> m_writtenStrings;
		DoneClasses m_doneClasses;
		int m_currentRememberTag;

		DoneObjects m_detailIndexFromObject; // object ptr -> index (used for backrefs and index in m_details)
		hkArray<hkUint8>::Temp m_bitfieldBuf; // all objects share same buffer for bitfield
		hkArray<ObjectDetails>::Temp m_details; // members present bitfield etc
		ListenerCache m_listenerCache;
		hkOstreamLogger* m_log;
		hkBool m_writeSerializeIgnored;
	};
}

hkResult hkBinaryTagfileWriter::save( const hkDataObject& obj, hkStreamWriter* stream, hkTagfileWriter::AddDataObjectListener* userListener, const Options& options )
{
 	Writer writer(stream, userListener, options);
	writer.writeObjectTree(obj);
	stream->flush();
	return stream->isOk() ? HK_SUCCESS : HK_FAILURE;
}

// explicitly instantiate our map type
#include <Common/Base/Container/PointerMap/hkMap.cxx>
template class hkMapBase<hkDataObject::Handle, int, Writer::HandleOps>;
template class hkMap<hkDataObject::Handle, int, Writer::HandleOps>;

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
