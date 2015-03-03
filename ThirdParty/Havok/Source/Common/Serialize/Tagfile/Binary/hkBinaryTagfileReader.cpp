/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Tagfile/Binary/hkBinaryTagfileCommon.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/System/Io/IArchive/hkIArchive.h>
#include <Common/Serialize/Data/hkDataObject.h>
#include <Common/Serialize/Tagfile/Binary/hkBinaryTagfileReader.h>
#include <Common/Serialize/Tagfile/Binary/hkTagStringList.h>

#include <Common/Base/Reflection/Util/hkVariantDataUtil.h>

// See hkBinaryTagfileWriter.cpp for comments on the various tagfile versions

#if 0
#include <Common/Base/DebugUtil/Logger/hkOstreamLogger.h>
#define LOG(A) if(m_log) m_log->A
#define LOG_INIT(FNAME) m_log = new hkOstreamLogger(FNAME)
#define LOG_QUIT() delete m_log
#else
class hkOstreamLogger;
#define LOG(A)
#define LOG_INIT(FNAME)
#define LOG_QUIT()
#endif

static inline hkUint32 convertEndianU32( hkUint32 n )
{
	union fmtU32
	{
		hkUint8 b[4];
		hkUint32 v;
	};
	union fmtU32 fDataIn, fDataOut;

	fDataIn.v = n;
	fDataOut.b[0] = fDataIn.b[3];
	fDataOut.b[1] = fDataIn.b[2];
	fDataOut.b[2] = fDataIn.b[1];
	fDataOut.b[3] = fDataIn.b[0];
	return fDataOut.v;
}

namespace
{
	struct IntReader
	{
		IntReader(hkIArchive* archive, int size):
			m_remaining(size),
			m_archive(archive)
		{
			m_buffer.setSize(size);
			m_buffer.setSizeUnchecked(0);
			m_cur = m_buffer.begin();
			m_end = m_buffer.end();
		}
		hkUint8 _fillBuffer()
		{
			HK_ASSERT(0x2343a244, m_remaining >= 0);
			m_buffer.setSizeUnchecked(m_remaining);
			m_archive->readArray8u(m_buffer.begin(), m_remaining);

			m_cur = m_buffer.begin();
			m_end = m_buffer.end();
			return *m_cur++;
		}
		HK_FORCE_INLINE hkUint8 readUint8()
		{
			return (m_cur < m_end) ? *m_cur++ : _fillBuffer();
		}
		HK_FORCE_INLINE hkInt32 readInt()
		{
			hkUint32 b = readUint8();
			hkUint32 u = (b & ~0x80) >> 1;
			hkBool32 neg = unsigned(b) & 1;
			unsigned shift = 6;
			while( b & 0x80 )
			{
				b = readUint8();
				u |= (b & ~0x80) << shift;
				shift += 7;
			}
			// We've got one, so one less remaining...
			m_remaining--;
			return neg ? -hkInt32(u) : hkInt32(u);
		}

		void readArray(hkDataArray& arr)
		{
			hkArray<hkInt32>::Temp intArray;
			const int size = m_remaining;

			intArray.setSize(size);
			for (int i = 0; i < size; i++)
			{
				intArray[i] = readInt();
			}
			arr.setAll(intArray.begin(), size);
			HK_ASSERT(0x3242a423, isAtEnd());
		}

		HK_FORCE_INLINE hkBool isAtEnd() const 
		{
			return m_remaining == 0 && m_cur == m_end;
		}

		int m_remaining;		
		hkIArchive* m_archive;
		const hkUint8* m_cur;
		const hkUint8* m_end;
		hkArray<hkUint8>::Temp m_buffer;
	};

		// see comments in hkBinaryTagfileWriter
	struct Reader
	{
		typedef hkDataObjectImpl ObjectImpl;
		typedef hkDataClassImpl ClassImpl;
		typedef hkPointerMap<int, ObjectImpl*> ReadObjects;
		typedef hkArray<ClassImpl*> ReadClasses;

		Reader(hkStreamReader* sr, hkDataWorld* cont)
			: m_tagfileVersion(-1)
			, m_ia(sr)
			, m_world(cont)
			, m_log(HK_NULL)
			, m_numPrevStringsStatic(0)
		{
			m_classes.pushBack(HK_NULL);
			m_prevStrings.pushBack(const_cast<char*>("")); // 0
			m_prevStrings.pushBack(HK_NULL); // -1
			m_numPrevStringsStatic = 2;
			LOG_INIT("tlogr.txt");

			//m_objCount = 0;
		}

		~Reader()
		{
			LOG_QUIT();
				// first 2 are const
			for( int i = m_numPrevStringsStatic; i < m_prevStrings.getSize(); ++i )
			{
				hkDeallocate<char>( m_prevStrings[i] );
			}
		}

		hkResult readHeader(hkBinaryTagfile::Header& out)
		{
			const hkInt32 magic1 = m_ia.read32();
			if(!m_ia.isOk()) { return HK_FAILURE; }
			const hkInt32 magic2 = m_ia.read32();
			if(!m_ia.isOk()) { return HK_FAILURE; }

			if(hkBinaryTagfile::isBinaryMagic(magic1, magic2))
			{
				out.m_magic1 = magic1;
				out.m_magic2 = magic2;
				out.m_fileInfo = readInt<int>();
				if(!m_ia.isOk()) { return HK_FAILURE; }

				out.m_version = readInt<int>();
				if(!m_ia.isOk()) { return HK_FAILURE; }

				switch(out.m_version)
				{
					case 0:
					case 1:
					case 2:
					case 3: // 3 is the same as 2 but with array size hints
						{
							break;
						}

					
					case 4: // 4 adds an sdk version string to the header
						{
							// read version 4
							// todo: support versioned string list
							HK_ASSERT(0x2b7ba2cc, m_rememberedObjects.getSize() == 0 );
							m_rememberedObjects.pushBack(HK_NULL);
							out.m_sdk = readString(); // only used for humans

							break;
						}
					default:
						{
							HK_WARN_ALWAYS(0x2ab6036f, "Unrecognised tagfile version " << m_tagfileVersion);
							return HK_FAILURE;
						}
				}

				return HK_SUCCESS;
			}

			return HK_FAILURE;
		}

		HK_FORCE_INLINE hkUint64 readInt64()
		{
			hkUint64 b = m_ia.read8u();
			hkUint64 u = (b & ~0x80) >> 1;
			hkBool32 neg = unsigned(b) & 1;
			unsigned shift = 6;
			while( b & 0x80 )
			{
				b = m_ia.read8u();
				u |= (b & ~0x80) << shift;
				shift += 7;
			}
			return  neg ? -hkInt64(u) : hkInt64(u);
		}
		HK_FORCE_INLINE hkUint64 readUint64()
		{
			return hkUint64(readInt64());
		}

		template<typename T>
		HK_FORCE_INLINE T readInt()
		{
			hkUint32 b = m_ia.read8u();
			hkUint32 u = (b & ~0x80) >> 1;
			hkBool32 neg = unsigned(b) & 1;
			unsigned shift = 6;
			while( b & 0x80 )
			{
				b = m_ia.read8u();
				u |= (b & ~0x80) << shift;
				shift += 7;
			}
			hkInt32 s = neg ? -hkInt32(u) : hkInt32(u);
			return T(s);
		}

		/* 
		template<>
		HK_FORCE_INLINE hkUint64 readInt<hkUint64>()
		{
			return hkUint64(readInt64());
		}
		template<>
		HK_FORCE_INLINE hkInt64 readInt<hkInt64>()
		{
			return readInt64();
		} */

		const char* readString()
		{
			int len = readInt<int>();
			if( len > 0 )
			{
				char* s = hkAllocate<char>( len + 1, HK_MEMORY_CLASS_SERIALIZE );
				m_ia.readRaw( s, len );
				s[len] = 0;
				m_prevStrings.pushBack(s);
				return s;
			}
			else
			{
				return m_prevStrings[-len]; // index starts at -2. 0 and -1 are reserved for ["", HK_NULL]
			}
		}

		float readFloat()
		{
			return m_ia.readFloat32();
		}

		void readBitfield( int numMembers, hkLocalArray<hkUint8>& bout)
		{
			int maxBits = HK_NEXT_MULTIPLE_OF(8, numMembers);
			int bitFieldNumBytes = maxBits / 8;
			bout.setSize(maxBits);
			hkUint8 b[64]; HK_ASSERT(0x17213b79, bitFieldNumBytes < (int)sizeof(b));
			m_ia.readArray8u( b, bitFieldNumBytes );
			for( int byteIndex = 0; byteIndex < bitFieldNumBytes; ++byteIndex )
			{
				unsigned curb = b[byteIndex];
				for( int i = 0; i < 8; ++i)
				{
					bout[byteIndex*8 + i] = hkUint8(curb & 1);
					curb >>= 1;
				}
			}
			for( int i = numMembers; i < maxBits; ++i )
			{
				HK_ASSERT(0x6c2c0450, bout[i] == 0 );
			}
			bout.setSize(numMembers);
		}

#if 0
		void _readArrayInts(hkDataArray& arr, int size)
		{
			for( int i = 0; i < size; ++i )
			{
				int val = readInt<int>();			
				arr[i] = val;
			}
		}
#else
		void _readArrayInts(hkDataArray& arr, int size)
		{
			if (size > 0)
			{
				IntReader reader(&m_ia, size);
				reader.readArray(arr);
			}
		}
#endif

		void _readArrayItems(hkDataArray& arr, int nelem)
		{
			hkDataObject::Type type = arr.getType();
			switch( type->getSubType())
			{
				case hkTypeManager::SUB_TYPE_BYTE:
				{
					hkArray<hkUint8>::Temp buf(nelem); buf.reserve(nelem);
					m_ia.readArray8u( buf.begin(), nelem );
					arr.setAll( buf.begin(), nelem );
					break;
				}
				case hkTypeManager::SUB_TYPE_INT:
				{
					// Pre-version 3, these were serialized as 32 bits anyway
					int sizeHint = hkSizeOf(hkInt32);
					if(m_tagfileVersion >= 3)
					{
						sizeHint = readInt<int>();
					}
					
					if(sizeHint == hkSizeOf(hkInt32))
					{
						_readArrayInts(arr, nelem);
					}
					else
					{
						for( int i = 0; i < nelem; ++i )
						{
							hkInt64 val = readInt64();
							arr[i] = val;
						}
					}
					break;
				}
				case hkTypeManager::SUB_TYPE_REAL:
				{
					hkArray<hkReal>::Temp tmp(nelem); tmp.setSize(nelem);
					m_ia.readArrayFloat32(tmp.begin(), tmp.getSize());
					arr.setAll( tmp.begin(), tmp.getSize() );
					break;
				}
				case hkTypeManager::SUB_TYPE_TUPLE:
				{
					if (type->isVec())
					{
						// This is a bit of a hack. The hkDataArrayDict - underlying implementation may not 
						// support setAll - for example with tuple/tuples producing an ArrayOfTuplesImplementation
						// Only a suitable dict representation implements asStridedBasicArray.. and thus the test
						hkStridedBasicArray stridedArray;
						hkBool canfastRead = (arr.asStridedBasicArray(stridedArray) == HK_SUCCESS);

						if (canfastRead)
						{
							const int numReal = type->getTupleSize();

							int numReadReal = numReal;
							if (numReal == 4)
							{
								numReadReal = readInt<int>();
								HK_ASSERT(0x234a324a, numReadReal == 4 || numReadReal == 3);
							}
							
							// Read everything
							hkArray<hkReal>::Temp buffer;
							buffer.setSize(nelem * numReal);
							m_ia.readArrayFloat32(buffer.begin(), nelem * numReadReal);

							// Okay I may need to go back and fix things...
							if (numReal != numReadReal)
							{
								HK_ASSERT(0x2423a43a, numReadReal == 3 && numReal == 4);

								const hkReal* src = buffer.begin() + (numReadReal * (nelem - 1));
								hkReal* dst = buffer.begin() + (numReal * (nelem - 1));
								for (int i = 0; i < nelem; i++)
								{
									dst[3] = 0.0f;
									dst[2] = src[2];
									dst[1] = src[1];
									dst[0] = src[0];

									src -= 3;
									dst -= 4;
								}
							}
							arr.setAll(buffer.begin(), nelem);
						}
						else
						{							
							const int numReal = type->getTupleSize();
							hkReal r[16];
							HK_ASSERT(0x2432432a, numReal <= (int)HK_COUNT_OF(r));

							int numReadReal = numReal;
							if (numReal == 4)
							{
								// Need to initialize buffer, in case we are reading 3, into vec4s
								r[3] = 0.0f;
								numReadReal = readInt<int>();
								HK_ASSERT(0x234a324a, numReadReal == 4 || numReadReal == 3);
							}
							
							for( int item = 0; item < nelem; ++item )
							{
								m_ia.readArrayFloat32(r, numReadReal);
								arr[item].setVec(r, numReal); // todo batch
							} 

						}
						break;
					}
					HK_ASSERT(0x23432432, !"Unhandled type");
					break;
				}
				case hkTypeManager::SUB_TYPE_POINTER:
				{
					for( int i = 0; i < nelem; ++i )
					{
						readObjectIntoValue(arr[i]);
					}
					break;
				}
				case hkTypeManager::SUB_TYPE_CSTRING:
				{
					for( int i = 0; i < nelem; ++i )
					{
						const char* s = readString();
						LOG(debug("String \"%s\"", s));
						arr[i] = s;
					}
					break;
				}
				case hkTypeManager::SUB_TYPE_CLASS:
				{
					arr.setSize(nelem);
					hkDataClass sclass = arr.getClass();
					int numMembers = sclass.getNumMembers();
					hkLocalArray<hkUint8> membersPresent(128); 
					readBitfield( numMembers, membersPresent );
					hkLocalArray<hkDataClass::MemberInfo> minfos(numMembers); minfos.setSizeUnchecked(numMembers);
					sclass.getAllMemberInfo(minfos);

					for( int memberIndex = 0; memberIndex < numMembers; ++memberIndex )
					{
						const hkDataClass::MemberInfo& mi = minfos[memberIndex];
						if( membersPresent[memberIndex] )
						{
							LOG(pushScope(mi.m_name));
							hkDataArray d = arr.swizzleObjectMember(mi.m_name);
							hkDataObject::Type memType = mi.m_type;

							if( memType->isTuple() && !memType->isVec())
							{
								for( int i = 0; i < nelem; ++i )
								{
									hkDataArray t = d[i].asArray();
									_readArrayItems( t, memType->getTupleSize());
								}
							}
							else
							{
								_readArrayItems(d, nelem);
							}
							LOG(popScope());
						}
					}
					break;
				}
				case hkTypeManager::SUB_TYPE_ARRAY:
				{	
					arr.setSize(nelem);

					for (int i = 0; i < nelem; i++)
					{
						const int childSize = readInt<int>();

						hkDataArray childArr = arr[i].asArray();
						childArr.setSize(childSize);

						_readArrayItems(childArr, childSize);
					}

					break;
				}
				default:
				{
					HK_ASSERT(0x618f9194, 0);
				}
			}
		}

		void readBinaryValue( hkDataObject::Value value, const hkDataClass::MemberInfo& minfo )
		{
			hkDataObject obj(value.m_impl);

			hkDataObject::Type type = minfo.m_type;
			if( type->isTuple() && !type->isVec())
			{
				hkDataArray arr( value.asArray());

				LOG(debug(type->asString()));
				_readArrayItems( arr, type->getTupleSize());
				return;
			}
			else if( type->isArray())
			{
				hkTypeManager& typeManager = m_world->getTypeManager();
				hkDataClass::MemberInfo _minfo = minfo;
				int asize = readInt<int>();

				hkDataObject::Type valueType = value.getType();
				hkDataObject::Type parentType = valueType->getParent();
				if (parentType->isClass() && parentType->getTypeName() == HK_NULL)
				{
					int aclass = readInt<int>();
					ClassImpl* cls = m_classes[aclass];

					_minfo.m_type = typeManager.makeArray(typeManager.addClass(cls->getName()));
				}

				hkDataArray arr( m_world->newArray(obj, value.m_handle, _minfo) );
				LOG(debug(arr.getType()->asString()));
				arr.setSize( asize );
				_readArrayItems( arr, asize );
				return;
			}

			switch( type->getSubType())
			{
				case hkTypeManager::SUB_TYPE_BYTE:
				{
					int v = m_ia.read8u();
					LOG(debug("Byte %x", v));
					value = v;
					break;
				}
				case hkTypeManager::SUB_TYPE_INT:
				{
					hkInt64 v = readInt64();
					LOG(debug("Int %x", v));
					value = v;
					break;
				}
				case hkTypeManager::SUB_TYPE_REAL:
				{
					hkReal r = readFloat();
					LOG(debug("Real %f", r));
					value = r;
					break;
				}
				case hkTypeManager::SUB_TYPE_TUPLE:
				{
					if (type->isVec())
					{
						const int numReal = type->getTupleSize();
						hkReal r[16];
						HK_ASSERT(0x3243a423, numReal <= (int) HK_COUNT_OF(r));

						LOG(debug("Vector%i", numReal));

						for( int i = 0; i < numReal; ++i )
						{
							r[i] = readFloat();
						}
						value.setVec(r, numReal);
						break;
					}
					HK_ASSERT(0x242432aa, !"Unhandled tuple type");
					break;
				}
				case hkTypeManager::SUB_TYPE_CLASS:
				{
					if( m_tagfileVersion >= 2 )
					{
						hkDataClassImpl* clsImpl = m_world->findClass(type->getTypeName());
						HK_ASSERT(0x32432423, clsImpl);

						// struct handling optimized in version 2+
						value = readObjectTopLevel(hkBinaryTagfile::TAG_OBJECT, clsImpl);
						break;
					}
					// Old version falls through and structs handled like objects
				}
				case hkTypeManager::SUB_TYPE_POINTER:
				{
					readObjectIntoValue(value);
					break;
				}
				case hkTypeManager::SUB_TYPE_CSTRING:
				{
					const char* s = readString();
					LOG(debug("String \"%s\"", s));
					value = s;
					break;
				}
				default:
				{
					HK_ASSERT(0x4c3c5273, 0);
				}
			}
		}

			// Read an object and assign it to an lvalue.
			// The value may be an array item or object member reference.
		template<typename ArrayOrObjectValue>
		void readObjectIntoValue(ArrayOrObjectValue value)
		{
				// Newer tagfile versions use a nonrecursive traversal when
				// saving and thus may refer to objects which haven't been saved yet.
			if( m_tagfileVersion >= 2 )
			{
				int id = readInt<int>();
				LOG(debug("Object *%08i", id));
				if( id < m_rememberedObjects.getSize() ) // yes, seen it
				{
					value = m_rememberedObjects[id];
				}
				else // not seen it yet, assign later when we see it
				{
					ForwardReferenceMap::Iterator it = m_forwardReferences.findOrInsertKey( id, HK_NULL );
					ForwardReferences* refs = m_forwardReferences.getValue(it);
					if( refs == HK_NULL )
					{
						refs = new ForwardReferences();
						m_forwardReferences.setValue(it, refs);
					}
					refs->remember(value);
				}
			}
			else // older tagfile version, fully recursive
			{
				LOG(debug("Object"));
				hkDataObject o = readObjectTopLevel(hkBinaryTagfile::TAG_NONE);
				value = o;
			}
		}

		hkDataObject readObjectTopLevel
			( hkBinaryTagfile::TagType objectTag
			, const hkDataClassImpl* klassImpl=HK_NULL )
		{
			if( objectTag == hkBinaryTagfile::TAG_NONE ) // fetch tag if not supplied
			{
				objectTag = readInt<hkBinaryTagfile::TagType>();
			}
			if( objectTag == hkBinaryTagfile::TAG_OBJECT_BACKREF )
			{
				int id = readInt<int>();
				LOG(debug("OBJ backref #%08i",id));
				return hkDataObject( m_rememberedObjects[id] );
			}
			else if( objectTag == hkBinaryTagfile::TAG_OBJECT_NULL )
			{
				LOG(debug("OBJ null"));
				return hkDataObject(HK_NULL);
			}

			hkDataClass klass( klassImpl
				? const_cast<hkDataClassImpl*>(klassImpl) // supplied class
				: m_classes[readInt<int>()] ); // else read one
			int numMembers = klass.getNumMembers();
			
			hkDataObject obj = m_world->newObject( klass );
			switch( objectTag )
			{
				case hkBinaryTagfile::TAG_OBJECT:
					break;
				case hkBinaryTagfile::TAG_OBJECT_REMEMBER:
				{
					int objectId = m_rememberedObjects.getSize();
					LOG(debug("Obj %s Remembered as &%08i", klass.getName(), objectId));
					m_rememberedObjects.pushBack( obj.getImplementation() );
					ForwardReferenceMap::Iterator it = m_forwardReferences.findKey(objectId);
					if( m_forwardReferences.isValid(it) )
					{
						ForwardReferences* ref = m_forwardReferences.getValue(it);
						ref->assign(obj);
						delete ref;
						m_forwardReferences.remove(it);
					}
					break;
				}
				default:
					HK_ERROR(0x484994d5, "corrupt file");
			}

			LOG(pushScope("Members"));
			hkLocalArray<hkUint8> membersPresent(128); 
			readBitfield(numMembers, membersPresent);
			hkLocalArray<hkDataClass::MemberInfo> minfos(numMembers); minfos.setSizeUnchecked(numMembers);
			klass.getAllMemberInfo(minfos);

			hkLocalArray<hkDataObject::MemberHandle> memHandles(numMembers); memHandles.setSizeUnchecked(numMembers);
			obj.getAllMemberHandles(memHandles);

			for( int memberIndex = 0; memberIndex < numMembers; ++memberIndex )
			{
				const hkDataClass::MemberInfo& mi = minfos[memberIndex];
				if( membersPresent[memberIndex] )
				{
					hkDataObject::Value value(obj.getImplementation(), memHandles[memberIndex]); 

					LOG(pushScope(mi.m_name));
					readBinaryValue( value, mi );
					LOG(popScope());
				}
				else
				{
					LOG(debug("Skip %s", mi.m_name));
				}
			}
			LOG(popScope());
			return obj;
		}

		hkResult readClass()
		{
			hkDataClass::Cinfo cinfo;
			cinfo.name = readString();
			cinfo.version = readInt<int>();
			if( m_world->getType() == hkDataWorld::TYPE_NATIVE )
			{
				HK_ASSERT2(0x530f818c, false, "hkDataWorldNative is not supported to load tagfile. Use hkDataWorldDict instead.");
				return HK_FAILURE;
			}
			int parentIndex = readInt<int>();
			cinfo.parent = ( parentIndex >= 0 && m_classes[parentIndex] ) ? m_classes[parentIndex]->getName() : HK_NULL;
			cinfo.members.setSize( readInt<int>() );
			LOG(debug("class %s %i members", cinfo.name, cinfo.members.getSize()));
			for( int i = 0; i < cinfo.members.getSize(); ++i )
			{
				hkDataClass::Cinfo::Member& m = cinfo.members[i];
				m.name = readString();
				const hkLegacyType::Type legacyType = readInt<hkLegacyType::Type>();
				const int tupleCount = ( legacyType & hkLegacyType::TYPE_TUPLE )
					? readInt<int>() 
					: 0;
				int bt = legacyType & hkLegacyType::TYPE_MASK_BASIC_TYPES;
				const char* className = ( bt == hkLegacyType::TYPE_STRUCT || bt == hkLegacyType::TYPE_OBJECT )
					? readString()
					: HK_NULL;
				if( className )
				{
					if( m_world->findClass(className) == HK_NULL )
					{
						HK_ASSERT3(0x3f64fd57, m_world->getType() != hkDataWorld::TYPE_NATIVE,
							"The class " << className << " is not registered in the provided native data world.\n"\
							"Did you set class name and type name registries for the world?");
					}
				}
				// Work out the type
				hkDataObject::Type type = m_world->getTypeManager().getType(legacyType, className, tupleCount);
				HK_ASSERT(0x23432432, type && "Legacy type needs to convert to type manager type.");

				
				m.type = type;

				// default?
			}
			m_classes.pushBack( m_world->newClass(cinfo) );
			return HK_SUCCESS;
		}

		hkDataObject readObjectTree()
		{
			
			{
				int magic0 = m_ia.read32();
				int magic1 = m_ia.read32();
				if( (magic0 != int(hkBinaryTagfile::BINARY_MAGIC_0) || magic1 != int(hkBinaryTagfile::BINARY_MAGIC_1))
	// 				&& (convertEndianU32(magic0) != hkUint32(hkBinaryTagfile::BINARY_MAGIC_0) || convertEndianU32(magic1) != hkUint32(hkBinaryTagfile::BINARY_MAGIC_1)) 
				  )
				{
					HK_WARN_ALWAYS(0x2ab6036f, "This does not look like a binary tagfile (magic number mismatch)");
					return hkDataObject(HK_NULL);
				}
	// 			m_ia.setByteSwap(convertEndianU32(magic0) == hkUint32(hkBinaryTagfile::BINARY_MAGIC_0));
			}
			for( hkBinaryTagfile::TagType tag = readInt<hkBinaryTagfile::TagType>();
				tag != hkBinaryTagfile::TAG_EOF;
				tag = readInt<hkBinaryTagfile::TagType>() )
			{
				switch( tag )
				{
					case hkBinaryTagfile::TAG_FILE_INFO:
					{
						m_tagfileVersion = readInt<int>();
						LOG(debug("header version %i", m_tagfileVersion));
						switch( m_tagfileVersion )
						{
							case 0:
							{
								break;
							}
							case 1:
							{
								int tagListVersion = readInt<int>();

								if(tagListVersion != HK_TAG_LIST_VERSION)
								{
									HK_ASSERT2(0x557f6908, 0, "Unknown Taglist version in Tagfile");
									return hkDataObject(HK_NULL);
								}
								m_prevStrings.clear();
								m_prevStrings.pushBack(const_cast<char*>("")); // 0
								m_prevStrings.pushBack(HK_NULL); // -1
								m_prevStrings.append(const_cast<char * const *>(HK_TAG_STRING_LIST), HK_NUM_TAG_STRINGS);
								m_numPrevStringsStatic = 2 + HK_NUM_TAG_STRINGS; // These elements are statically allocated, subsequent elements are freed on destruction
								break;
							}
							case 2:
							case 3: // 3 is the same as 2 but with array size hints
							case 4: // 4 adds an sdk version string to the header
							{
								// todo: support versioned string list
								HK_ASSERT(0x2b7ba2cc, m_rememberedObjects.getSize() == 0 );
								m_rememberedObjects.pushBack(HK_NULL);
								if( m_tagfileVersion == 4 )
								{
									readString(); // only used for humans
								}
								break;
							}
							default:
							{
								HK_WARN_ALWAYS(0x2ab6036f, "Unrecognised tagfile version " << m_tagfileVersion);
								return hkDataObject(HK_NULL);
							}
						}
						break;
					}
					case hkBinaryTagfile::TAG_METADATA:
					{
						if( readClass() == HK_FAILURE )
						{
							return hkDataObject(HK_NULL);
						}
						break;
					}
					case hkBinaryTagfile::TAG_OBJECT:
					case hkBinaryTagfile::TAG_OBJECT_REMEMBER:
					case hkBinaryTagfile::TAG_OBJECT_NULL:
					{
						hkDataObject o = readObjectTopLevel(tag);
						if( m_tagfileVersion < 2 ) 
						{
							// The old fully recursive way.
							// When the Top object is read, we're done
							return o;
						}
						//m_objCount++;

						// New style, multiple toplevels, loop until TAG_FILE_END
						break;
					}
					case hkBinaryTagfile::TAG_FILE_END:
					{
						HK_ASSERT2(0x32b2434a, m_forwardReferences.getSize() == 0, "There are still dangling references");
						HK_ASSERT2(0x32b2434b, m_rememberedObjects.getSize() >= 2, "No objects found in this file?");
						return m_rememberedObjects[1]; // first object is always null, seconds is contents
					}
					default:
					{
 						HK_ASSERT(0x1aa61754, 0);
						return hkDataObject(HK_NULL);
					}
				}
			}
			return hkDataObject(HK_NULL);
		}

		int m_tagfileVersion;
		hkIArchive m_ia;
		hkDataWorld* m_world;
		hkOstreamLogger* m_log;
		ReadClasses m_classes;
		hkArray<char*> m_prevStrings;
		int m_numPrevStringsStatic;
		hkArray<hkDataObjectImpl*> m_rememberedObjects;

		//int m_objCount;

		struct ForwardReferences
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_SERIALIZE, ForwardReferences);
			void remember( hkDataObject::Value value)
			{
				m_objectRefs.pushBack(value);
			}
			void remember( hkDataArray::Value value)
			{
				m_arrayRefs.pushBack(value);
			}
			void assign( hkDataObject& o )
			{
				for( int i = 0; i < m_arrayRefs.getSize(); ++i )
				{
					m_arrayRefs[i] = o;
				}
				for( int i = 0; i < m_objectRefs.getSize(); ++i )
				{
					m_objectRefs[i] = o;
				}
			}

			hkArray<hkDataArray::Value> m_arrayRefs;
			hkArray<hkDataObject::Value> m_objectRefs;
		};
		typedef hkPointerMap<int, ForwardReferences*> ForwardReferenceMap;
		ForwardReferenceMap m_forwardReferences;
	};

} // namespace anonymous

hkBinaryTagfileReader::hkBinaryTagfileReader()
{
}

hkDataObject hkBinaryTagfileReader::load( hkStreamReader* stream, hkDataWorld& cont )
{
	Reader reader(stream, &cont);
	return reader.readObjectTree();
}

hkResult hkBinaryTagfileReader::readHeader(hkStreamReader* stream, hkBinaryTagfile::Header& out)
{
	Reader reader(stream, HK_NULL);
	return reader.readHeader(out);
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
