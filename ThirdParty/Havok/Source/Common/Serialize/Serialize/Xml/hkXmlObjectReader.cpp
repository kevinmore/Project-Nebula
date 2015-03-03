/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Serialize/Xml/hkXmlObjectReader.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/System/Io/IStream/hkIStream.h>
#include <Common/Base/System/Io/Reader/hkStreamReader.h>
#include <Common/Serialize/Serialize/hkRelocationInfo.h>
#include <Common/Serialize/Util/Xml/hkXmlParser.h>
#include <Common/Base/Container/RelArray/hkRelArray.h>

namespace
{
	struct XmlObjectReader_DummyArray
	{
		void* ptr;
		int size;
		int cap;
	};

	struct XmlObjectReader_DummyHomogeneousArray
	{
		const hkClass* klass;
		void* data;
		int size;
		//              int capAndFlags;
	};

	template <typename T>
	T& lookupMember(void* start)
	{
		return *reinterpret_cast<T*>( start );
	}

	struct Buffer
	{
		public:

			enum Pad
			{
				PAD_NONE = 1,
				PAD_4 = 4,
				PAD_8 = 8,
				PAD_16 = 16
			};

			Buffer(hkArray<char>& c)
				:	m_buf(c)
			{
			}

				// Reserve space for nbytes - fill with zeros
				// Return offset of writable space
			int reserve(int nbytes, Pad pad = PAD_NONE)
			{
				int orig = m_buf.getSize();
				int size = HK_NEXT_MULTIPLE_OF(pad, orig + nbytes);
				m_buf.setSize( size, 0 );
				m_buf.setSizeUnchecked( orig );
				return orig;
			}

				// Advance by nbytes - should never cause reallocation
				// because we should have previously reserve()d space.
			void advance( int nbytes, Pad pad = PAD_NONE )
			{
				int size = HK_NEXT_MULTIPLE_OF(pad, m_buf.getSize() + nbytes);
				HK_ASSERT2(0x12402f4c, size <= m_buf.getCapacity(), "Overflowing XML write buffer capacity, will cause resize." );
				m_buf.setSizeUnchecked( size );
			}

			void* pointerAt( int offset )
			{
				HK_ASSERT2(0x6c591289, offset <= m_buf.getSize(), "Offset of pointer not within XML buffer range." );
				return m_buf.begin() + offset;
			}

		private:

			hkArray<char>& m_buf;
	};

	void* addByteOffset(void* p, int n)
	{
		return static_cast<char*>(p) + n;
	}
}


hkXmlObjectReader::hkXmlObjectReader(hkXmlParser* parser, hkStringMap<void*>* nameToObject)
	: m_parser(parser), m_nameToObject(nameToObject)
{
	if(parser)
	{
		m_parser->addReference();
	}
	else
	{
		m_parser = new hkXmlParser();
	}
}

hkXmlObjectReader::~hkXmlObjectReader()
{
	m_parser->removeReference();
}

static inline hkResult extractCstring(int memberStartOffset, const hkStringBuf& text, Buffer& buffer, hkRelocationInfo& reloc)
{
	if( text.getLength() != 1 || text[0] != 0 ) // null pointer is represented as single null
	{
		int len = text.getLength() + 1; // include null
		int textOffset = buffer.reserve( len, Buffer::PAD_16 );
		hkString::memCpy( buffer.pointerAt(textOffset), text.cString(), len );
		reloc.addLocal( memberStartOffset, textOffset );
		buffer.advance( len, Buffer::PAD_16 );
		return HK_SUCCESS;
	}
	return HK_FAILURE;
}

static inline hkBool isSpace(int c)
{
	return (c==' ') || (c =='\t') || (c=='\n') || (c=='\r');
}

static int getNumElementsInMember( const hkClassMember& member )
{
	int asize = member.getCstyleArraySize();
	return asize ? asize : 1;
}

// read a member into the location curp.
// The input data is available as an istream.
// Arrays are temporarily stored in array* for later processing.
static hkResult readSinglePodMember(hkClassMember::Type mtype, void* curp, hkIstream& is )
{
	switch( mtype ) // match order with order of hktypes
	{
		case hkClassMember::TYPE_BOOL:
		{
			hkBool* f = static_cast<hkBool*>(curp);
			is >> *f;
			break;
		}
		case hkClassMember::TYPE_CHAR:
		{
			char* f = static_cast<char*>(curp);
			is.read(f, 1);
			break;
		}
		case hkClassMember::TYPE_INT8:
		{
			hkInt8* f = static_cast<hkInt8*>(curp);
			int foo;
			is >> foo;
			*f = static_cast<hkInt8>(foo);
			break;
		}
		case hkClassMember::TYPE_UINT8:
		{
			hkUint8* f = static_cast<hkUint8*>(curp);
			int foo;
			is >> foo;
			*f = static_cast<hkUint8>(foo);
			break;
		}
		case hkClassMember::TYPE_INT16:
		{
			hkInt16* f = static_cast<hkInt16*>(curp);
			is >> *f;
			break;
		}
		case hkClassMember::TYPE_UINT16:
		{
			hkUint16* f = static_cast<hkUint16*>(curp);
			is >> *f;
			break;
		}
		case hkClassMember::TYPE_INT32:
		{
			hkInt32* f = static_cast<hkInt32*>(curp);
			is >> *f;
			break;
		}
		case hkClassMember::TYPE_UINT32:
		{
			hkUint32* f = static_cast<hkUint32*>(curp);
			is >> *f;
			break;
		}
		case hkClassMember::TYPE_INT64:
		{
			hkInt64* f = static_cast<hkInt64*>(curp);
			is >> *f;
			break;
		}
		case hkClassMember::TYPE_UINT64:
		{
			hkUint64* f = static_cast<hkUint64*>(curp);
			is >> *f;
			break;
		}
		case hkClassMember::TYPE_ULONG:
		{
			hkUint64 tmpStorage;
            is >> tmpStorage;
			*static_cast<hkUlong*>(curp) = hkUlong(tmpStorage);
			break;
		}
		case hkClassMember::TYPE_REAL:
		{
			hkReal* f = static_cast<hkReal*>(curp);
			is >> *f;
			break;
		}
		case hkClassMember::TYPE_HALF:
		{
			hkHalf* hf = static_cast<hkHalf*>(curp);
			hkReal f;
			is >> f;
			hf->setReal<false>(f);
			break;
		}
		case hkClassMember::TYPE_VECTOR4:
		case hkClassMember::TYPE_QUATERNION:
		{
			hkReal* f = static_cast<hkReal*>(curp);
			is >> f[0] >> f[1] >> f[2] >> f[3];
			break;
		}
		case hkClassMember::TYPE_MATRIX3:
		case hkClassMember::TYPE_ROTATION:
		{
			hkReal* f = static_cast<hkReal*>(curp);
			hkString::memSet(f, 0, 12*sizeof(hkReal));
			is >> f[0] >> f[1] >> f[2];
			is >> f[4] >> f[5] >> f[6];
			is >> f[8] >> f[9] >> f[10];
			break;
		}
		case hkClassMember::TYPE_QSTRANSFORM:
		{
			hkReal* f = static_cast<hkReal*>(curp);
			is >> f[0] >> f[1] >> f[2];
			is >> f[4] >> f[5] >> f[6] >> f[7];
			is >> f[8] >> f[9] >> f[10];
			f[3] = f[11] = 0;
			break;
		}
		case hkClassMember::TYPE_MATRIX4:
		{
			hkReal* f = static_cast<hkReal*>(curp);
			is >> f[0] >> f[1] >> f[2] >> f[3];
			is >> f[4] >> f[5] >> f[6] >> f[7];
			is >> f[8] >> f[9] >> f[10] >> f[11];
			is >> f[12] >> f[13] >> f[14] >> f[15];
			break;
		}
		case hkClassMember::TYPE_TRANSFORM:
		{
			hkReal* f = static_cast<hkReal*>(curp);
			is >> f[0] >> f[1] >> f[2];
			is >> f[4] >> f[5] >> f[6] ;
			is >> f[8] >> f[9] >> f[10];
			is >> f[12] >> f[13] >> f[14];
			f[3] = f[7] = f[11] = 0;
			f[15] = 1;
			break;
		}
		default:
		{
			HK_ERROR(0x19fca9ad, "Class member unknown / unhandled: " << mtype );
		}
	}
	return is.isOk() ? HK_SUCCESS : HK_FAILURE;
}

static hkBool32 extractText(hkXmlParser& parser, hkStreamReader* reader, hkXmlParser::StartElement* start, bool canonicalize, hkStringBuf& ret)
{
	hkBool32 gotText = true;
	hkXmlParser::Node* node = HK_NULL;
	parser.nextNode(&node, reader);
	if( hkXmlParser::Characters* chars = node->asCharacters() )
	{
		if( canonicalize )
		{
			chars->canonicalize("(),");
		}
		ret = chars->text;
		gotText = chars->text.cString() != HK_NULL; // null string (not empty string "")
		delete node;
	}
	else if( node->asEnd() )
	{
		// empty characters ok == the empty string
		parser.putBack(node);
	}
	else
	{
		HK_ASSERT3(0x6cc04e1a, 0, "Parse error, expected characters after " << start->name );
	}
	return gotText;
}

static hkResult consumeEndElement(hkXmlParser& parser, hkStreamReader* reader, hkXmlParser::StartElement* start)
{
	hkResult retValue = HK_SUCCESS;
	hkXmlParser::Node* node = HK_NULL;
	parser.nextNode(&node, reader);
	hkXmlParser::EndElement* end = node->asEnd();
	if( end == HK_NULL || end->name != start->name )
	{
		HK_ASSERT3(0x4cadab61, end != HK_NULL, "Parse error, expected end element for " << start->name );
		HK_ASSERT3(0x2c0e93cf, end && (end->name == start->name), "Mismatched end element for " << start->name );
		retValue = HK_FAILURE;
	}
	delete node;
	return retValue;
}

static int loadSimpleArray(const hkClassMember& member, const hkStringBuf& text, Buffer& buffer, hkRelocationInfo& reloc)
{
	int numElements = 0;
	hkClassMember::Type mtype = member.getArrayType();
	switch( mtype )
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
		{
			int msize = member.getArrayMemberSize();
			hkIstream istream( text.cString(), text.getLength() );
			int off = buffer.reserve( msize );
			// Variable used to skip one character after every element in the array 
			// (required when reading an array of chars).
			char c; 
			while( readSinglePodMember(mtype, buffer.pointerAt(off), istream) == HK_SUCCESS )
			{
				numElements += 1;
				istream.get(c);
				// Either we are in the middle of the array or we should have exhausted the text line.
				HK_ASSERT(0x60ddf9fe, c == ' ' || !istream.isOk());
				buffer.advance( msize );
				off = buffer.reserve( msize );
			}
			break;
		}
		case hkClassMember::TYPE_ZERO:
		{
			break;
		}
		case hkClassMember::TYPE_VARIANT:
		{
			// fallthrough - arrays of variants are almost like
			// arrays of pointers except that the number of elements
			// needs to be halved at the end.
		}
		case hkClassMember::TYPE_POINTER:
		case hkClassMember::TYPE_FUNCTIONPOINTER:
		{
			int s = 0;
			for( int i = 0; i < text.getLength(); ++i )
			{
				if( isSpace(text[i]) )
				{
					if( s != i )
					{
						hkLocalArray<char> id( i - s + 1 );
						hkString::memCpy( id.begin(), text.cString()+s, i-s );
						id.setSize(i-s+1);
						id[i-s] = 0;
						int off = buffer.reserve( sizeof(void*) );
						reloc.addImport( off, id.begin() );
						buffer.advance( sizeof(void*) );
						numElements += 1;
					}
					s = i+1;
					continue;
				}
			}
			if( s != text.getLength() )
			{
				int off = buffer.reserve( sizeof(void*) );
				reloc.addImport( off, text.cString()+s );
				buffer.advance( sizeof(void*) );
				numElements += 1;
			}
			if( mtype == hkClassMember::TYPE_VARIANT )
			{
				HK_ASSERT2(0x3bfa0c3d, (numElements & 1) == 0, "Corrupt variant array");
				numElements /= 2;
			}
			break;
		}
		//case hkClassMember::TYPE_STRUCT:
		//case hkClassMember::TYPE_ARRAY:
		//case hkClassMember::TYPE_INPLACEARRAY:
		//case hkClassMember::TYPE_ENUM:
		//case hkClassMember::TYPE_SIMPLEARRAY:
		//case hkClassMember::TYPE_HOMOGENEOUSARRAY:
		//case hkClassMember::TYPE_FLAGS:
		//case hkClassMember::TYPE_RELARRAY:
		default:
		{
			// these aren't simple types
			HK_ASSERT2(0x6cc0400a, 0, "Load simple array called on a member that is not a simple array." );
		}
	}
	buffer.reserve(0, Buffer::PAD_16);
	buffer.advance(0, Buffer::PAD_16);

	return numElements;
}

static void skipDataForPartialReflected( const hkClass& klass, const hkClassMember* member, hkXmlParser& parser, hkXmlParser::StartElement* startElement, hkStreamReader* reader )
{
	int level = 1;
	hkBool32 warned = false;
	hkXmlParser::Node* n = HK_NULL;
	while( parser.nextNode( &n, reader ) == HK_SUCCESS && n != HK_NULL )
	{
		switch( n->type )
		{
			case hkXmlParser::START_ELEMENT:
			{
				++level;
				break;
			}
			case hkXmlParser::END_ELEMENT:
			{
				if( --level == 0 )
				{
					parser.putBack( n );
					return;
				}
				break;
			}
			case hkXmlParser::CHARACTERS:
			{
				if( ! warned )
				{
					HK_WARN(0x28cd8bfd, "Found data for non-reflected '"<<klass.getName()<<"::"<<member->getName()<<"'. Ignoring it.");
					warned = true;
				}
				break;
			}
			default:
			{
				HK_ASSERT(0x18cb354d, 0); // unreachable
			}
		}
		delete n;
		n = HK_NULL;
	}
	HK_WARN(0x28cd8bfe, "Parse error?");
}

static int readEmbeddedHomogeneousClass( hkXmlParser& parser, hkStreamReader* reader, hkXmlParser::StartElement* startElement, Buffer& localBuffer, Buffer& buffer, hkRelocationInfo& reloc, hkStringMap<void*>* nameToObject );

static hkResult readClassBody(
	const hkClass& klass,
	int classStartOffset,
	Buffer& buffer,
	hkXmlParser::StartElement* topElement,
	hkXmlParser& parser,
	hkStreamReader* reader,
	hkRelocationInfo& reloc,
	hkStringMap<void*>* nameToObject)
{
	hkXmlParser::Node* node;

	while( parser.nextNode(&node, reader) == HK_SUCCESS )
	{
		if( hkXmlParser::StartElement* startElement = node->asStart() )
		{
			HK_ASSERT3(0x3acc8f13, startElement->name == "hkparam", "XML element starts with " << startElement->name << ", not with the expected 'hkparam'." );
			const char* paramName  = startElement->getAttribute("name",HK_NULL);
			HK_ASSERT2(0x3bbb5581, paramName != HK_NULL, "XML element missing 'name' attribute." );

			const hkClassMember* member = klass.getMemberByName(paramName);
 			if( member == HK_NULL )
			{
				HK_WARN(0x28cd8bfc, "Unknown member '"<<klass.getName()<<"::"<<paramName<<"'. Ignoring it.");
				hkXmlParser::Tree tree;
				parser.expandNode(startElement, tree, reader);
				startElement->removeReference();
				// tree destructor deletes xml nodes.
				continue;
			}

			switch( member->getType() )
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
				{
					hkStringBuf text; extractText(parser, reader, startElement, true, text);
					hkIstream iss( text.cString(), text.getLength() );
						void* memberAddress = buffer.pointerAt( classStartOffset + member->getOffset() );
						int numElem = getNumElementsInMember(*member);
						int elemSize = member->getSizeInBytes() / numElem;
						for( int i = 0; i < numElem; ++i )
						{
							readSinglePodMember( member->getType(), memberAddress, iss );
							memberAddress = addByteOffset(memberAddress, elemSize);
						}
					break;
				}
				case hkClassMember::TYPE_ZERO:
				{
					break; // auto zeroed anyway
				}
				case hkClassMember::TYPE_CSTRING:
				case hkClassMember::TYPE_STRINGPTR:
				{
					hkStringBuf text;
					if( extractText(parser, reader, startElement, false, text) )
					{
						extractCstring(classStartOffset + member->getOffset(), text, buffer, reloc);
					}
					break;
				}
				case hkClassMember::TYPE_POINTER:
				case hkClassMember::TYPE_FUNCTIONPOINTER:
				{
					if( member->getType() == hkClassMember::TYPE_POINTER && member->getSubType() == hkClassMember::TYPE_CHAR )
					{
						hkStringBuf text;
						if( extractText(parser, reader, startElement, false, text) )
						{
							extractCstring(classStartOffset + member->getOffset(), text, buffer, reloc);
						}
					}
					else
					{
						hkStringBuf text; extractText(parser, reader, startElement, true, text);
						int curElement = 0;
						int maxElements = getNumElementsInMember(*member);
						int elemSize = member->getSizeInBytes() / maxElements;
						int memberOffset = classStartOffset + member->getOffset();
						int s = 0;
						for( int i = 0; curElement < maxElements && i < text.getLength(); ++i )
						{
							if( isSpace(text[i]) )
							{
								if( s != i )
								{
									hkLocalArray<char> id( i - s + 1 );
									hkString::memCpy( id.begin(), text.cString()+s, i-s );
									id.setSize(i-s+1);
									id[i-s] = 0;
									reloc.addImport( memberOffset + curElement * elemSize, id.begin() );
									++curElement;
								}
								s = i+1;
								continue;
							}
						}
						if( s != text.getLength() )
						{
							if( curElement < maxElements )
							{
								reloc.addImport( memberOffset + curElement * elemSize, text.cString()+s );
								++curElement;
							}
							else
							{
								HK_WARN(0x7635e7cf, "Too many initializers for " << klass.getName() << "::" << member->getName() );
							}
						}
					}
					break;
				}
				case hkClassMember::TYPE_ARRAY:
				case hkClassMember::TYPE_INPLACEARRAY:
				case hkClassMember::TYPE_SIMPLEARRAY:
				case hkClassMember::TYPE_RELARRAY:
				{
					int numElements = -1;
					int arrayBeginOffset = buffer.reserve(0);

					if( member->getSubType() == hkClassMember::TYPE_STRUCT )
					{
						const char* numElementsString = startElement->getAttribute("numelements",HK_NULL);
						HK_ASSERT2(0x3cbc5582, numElementsString != 0, "Could not find 'numelements' attribute in an array of structs.");
						numElements = hkString::atoi( numElementsString );
						if( const hkClass* sclass = member->getClass() )
						{
							int ssize = sclass->getObjectSize();
							buffer.reserve( ssize * numElements, Buffer::PAD_16 );
							buffer.advance( ssize * numElements, Buffer::PAD_16 );
							for( int i = 0; i < numElements; ++i )
							{
								hkXmlParser::Node* snode = HK_NULL;
								parser.nextNode(&snode, reader);
								readClassBody( *sclass, arrayBeginOffset + i*ssize, buffer,
									snode->asStart(), parser, reader, reloc, nameToObject );
								delete snode;
							}
						}
						else
						{
							skipDataForPartialReflected( klass, member, parser, startElement, reader );
						}
						// maybe peek and assert next is </hkparam>
					}
					else if( member->getSubType() == hkClassMember::TYPE_ENUM || member->getSubType() == hkClassMember::TYPE_FLAGS )
					{
						HK_ASSERT2(0x3cbc5583,0, "Arrays of enums/flags not supported in XML reader yet.");
						//XXX fixme
					}
					else if( member->getArrayType() == hkClassMember::TYPE_CSTRING || member->getArrayType() == hkClassMember::TYPE_STRINGPTR )
					{
						const char* numElementsString = startElement->getAttribute("numelements",HK_NULL);
						HK_ASSERT2(0x3cbc5582, numElementsString != 0, "Could not find 'numelements' attribute in an array of c-strings.");
						numElements = hkString::atoi( numElementsString );
						buffer.reserve( sizeof(char*) * numElements, Buffer::PAD_16 );
						buffer.advance( sizeof(char*) * numElements, Buffer::PAD_16 );
						for( int i = 0; i < numElements; ++i )
						{
							hkXmlParser::Node* snode = HK_NULL;
							parser.nextNode(&snode, reader);
							HK_ASSERT2(0x3acc8f13, snode->asStart() && snode->asStart()->name == "hkcstring", "Expected <hkcstring>" );

							hkStringBuf text;
							if( extractText(parser, reader, startElement, false, text) )
							{
								extractCstring(arrayBeginOffset + i*sizeof(char*), text, buffer, reloc);
							}
							// skip to next hkcstring
							delete snode;
							parser.nextNode(&snode, reader);
							HK_ASSERT2(0x3acc8f13, snode->asEnd() && snode->asEnd()->name == "hkcstring", "Expected </hkcstring>" );
							delete snode;
						}
					}
					else if( member->getSubType() == hkClassMember::TYPE_VOID )
					{
						skipDataForPartialReflected( klass, member, parser, startElement, reader );
						numElements = 0;
					}
					else
					{
						bool canonicalize = member->getSubType() != hkClassMember::TYPE_STRUCT;
						hkStringBuf text; extractText(parser, reader, startElement, canonicalize, text);
						numElements = loadSimpleArray(*member, text, buffer, reloc );
					}

					HK_ASSERT(0x7ede835a, numElements >= 0 );
					

					if( member->getType() !=  hkClassMember::TYPE_RELARRAY )
					{
						XmlObjectReader_DummyArray& dummy = lookupMember<XmlObjectReader_DummyArray>( buffer.pointerAt(classStartOffset + member->getOffset()) );
						dummy.ptr = HK_NULL;
						dummy.size = numElements;
						if( numElements > 0 )
						{
							reloc.addLocal( classStartOffset + member->getOffset(), arrayBeginOffset );
						}

						if( member->getType() != hkClassMember::TYPE_SIMPLEARRAY )
						{
							dummy.cap = numElements | hkArray<char>::DONT_DEALLOCATE_FLAG;
						}
					}
					else // it is an hkClassMember::TYPE_RELARRAY
					{
						hkRelArray<hkUint8>& dummy = lookupMember< hkRelArray<hkUint8> >( buffer.pointerAt(classStartOffset + member->getOffset()) );
						dummy._setSize( static_cast<hkUint16>(numElements) );
						dummy._setOffset(0);
						if( numElements > 0 )
						{
							// don't need relocation info for rel arrays, if the buffer is moved in memory the data will still be valid
							int offset = (arrayBeginOffset) - (classStartOffset + member->getOffset());
							HK_ASSERT(0x4d984ae5, offset > 0);
							dummy._setOffset( static_cast<hkUint16>(offset) );
						}
					}
					
					break;
				}
				case hkClassMember::TYPE_ENUM:
				{
					hkStringBuf text; extractText(parser, reader, startElement, false, text);
					if( text.getLength() )
					{
						const hkClassEnum& cenum = member->getEnumType();

						int val = 0;
						if( cenum.getValueOfName( text.cString(), &val) != HK_SUCCESS )
						{
							HK_WARN(0x555b54ab, "Invalid enum string '" << text.cString() << "' found for '"
								<< cenum.getName() << "' in member '" << member->getName() );
						}
						member->setEnumValue( buffer.pointerAt(classStartOffset + member->getOffset()), val);
					}
					break;
				}
				case hkClassMember::TYPE_STRUCT:
				{
					for( int i = 0; i < getNumElementsInMember(*member); ++i )
					{
						hkXmlParser::Node* snode = HK_NULL;
						parser.nextNode(&snode, reader);
						if( hkXmlParser::StartElement* structStart = snode->asStart() )
						{
							HK_ASSERT3(0x48b01b1e, structStart != HK_NULL && (structStart->name == "struct" || structStart->name == "hkobject"),
								"Parse error, expected <struct> or <hkobject> after " << startElement->name );
							if( const hkClass* sclass = member->getClass() )
							{
								int structOffset = classStartOffset + member->getOffset() + i * member->getStructClass().getObjectSize();
								readClassBody( member->getStructClass(), structOffset,
									buffer, structStart, parser, reader, reloc, nameToObject );
								delete snode;
							}
							else
							{
								skipDataForPartialReflected( klass, member, parser, structStart, reader );
							}
						}
						else
						{
							parser.putBack( snode );
							break;
						}
					}
					break;
				}
				case hkClassMember::TYPE_HOMOGENEOUSARRAY:
				{
					int numElements = -1;
					const char* numElementsString = startElement->getAttribute("numelements",HK_NULL);
					HK_ASSERT2(0x3cbc5582, numElementsString != HK_NULL, "Could not find 'numelements' attribute in homogenous array.");
					numElements = hkString::atoi( numElementsString );
					hkXmlParser::Node* snode = HK_NULL;
					HK_ON_DEBUG(hkResult res = )parser.nextNode(&snode, reader);
					HK_ASSERT3(0x076ed43d, res == HK_SUCCESS, "Xml parser error, expected homogeneous array data in node '" << startElement->name << "'.");
					HK_ASSERT3(0x076ed43e, (numElements > 0 && snode->type != hkXmlParser::END_ELEMENT)
											|| (numElements == 0 && snode->type == hkXmlParser::END_ELEMENT),
											"Xml parser error, expected homogeneous array of " << numElements << " items in node '" << startElement->name << "'.");
					parser.putBack(snode);
					if( snode->type == hkXmlParser::END_ELEMENT )
					{
						HK_ASSERT3(0x076ed43d, numElements == 0, "Xml parser error, expected homogeneous array data of " << numElements << " items in node '" << startElement->name << "'.");
						break;
					}
					HK_ASSERT3(0x076ed43e, numElements > 0, "Xml parser error, expect no homogeneous array data in node '" << startElement->name << "'.");
					// read homogeneous array class
					const hkClass* sclass = HK_NULL;
					// memory/buffer for embedded class
					hkLocalArray<char> localStorage(1024);
					Buffer localBuffer( localStorage );
					// Try to load the embedded class, file version < 6
					if( snode->type == hkXmlParser::START_ELEMENT )
					{
						int classOffset = readEmbeddedHomogeneousClass(parser, reader, startElement, localBuffer, buffer, reloc, nameToObject);
						sclass = static_cast<const hkClass*>( localBuffer.pointerAt(0) );
						// Add local for class
						reloc.addLocal( classStartOffset + member->getOffset(), classOffset );
					}
					else if( snode->type == hkXmlParser::CHARACTERS )
					{
						// read class pointer id, file version 6
						hkStringBuf id; extractText(parser, reader, startElement, true, id);
						HK_ASSERT(0x456de543, nameToObject && nameToObject->hasKey(id.cString()));
						sclass = static_cast<const hkClass*>( nameToObject->getWithDefault(id.cString(), HK_NULL) );
						reloc.addImport( classStartOffset + member->getOffset(), id.cString() );
					}
					else
					{
						HK_ASSERT3(0x076ed43f, false, "Xml parser error, found unknown node type. Expected homogeneous array data of " << numElements << " items in node '" << startElement->name << "'.");
					}
					int elemsize = sclass->getObjectSize();
					int arrayBeginOffset = buffer.reserve(0);
					buffer.reserve( elemsize * numElements, Buffer::PAD_16 );
					buffer.advance( elemsize * numElements, Buffer::PAD_16 );

					for( int i = 0; i < numElements; ++i )
					{
						hkXmlParser::Node* structnode = HK_NULL;
						parser.nextNode(&structnode, reader);
						readClassBody( *sclass, arrayBeginOffset + i*elemsize, buffer,
							structnode->asStart(), parser, reader, reloc, nameToObject );
						delete structnode;
					}

					XmlObjectReader_DummyHomogeneousArray& dummy = lookupMember<XmlObjectReader_DummyHomogeneousArray>( buffer.pointerAt(classStartOffset + member->getOffset()) );
					dummy.klass = HK_NULL;
					dummy.data = HK_NULL;
					dummy.size = numElements;
					if( numElements > 0 )
					{
						reloc.addLocal( classStartOffset + member->getOffset() +sizeof(hkClass*), arrayBeginOffset );
					}
					break;
				}
				case hkClassMember::TYPE_VARIANT:
				{
					hkStringBuf text; extractText(parser, reader, startElement, true, text);
					int space = text.indexOf(' ');
					if( space != -1 )
					{
						int off = classStartOffset + member->getOffset();
						hkStringBuf s = text;
						s.slice(0, space);
						reloc.addImport( off, s.cString() );
						s = text;
						s.chompStart(space+1);
						reloc.addImport( off+sizeof(void*), s.cString() );
					}
					break;
				}
				case hkClassMember::TYPE_FLAGS:
				{
					hkStringBuf stext; extractText(parser, reader, startElement, false, stext);
					if( stext.getLength() )
					{
						const hkClassEnum& cenum = member->getEnumType();
						hkArray<char> text; text.setSize( stext.getLength() + 1 ); //copy for modification
						hkString::strNcpy( text.begin(), stext.cString(), stext.getLength()+1 );
						char* cur = text.begin();
						int accum = 0;
						while( cur )
						{
							char* next = HK_NULL;
							if( char* bar = const_cast<char*>( hkString::strChr(cur, '|') ) )
							{
								*bar = 0; // bar = ptr to '|' separator
								next = bar+1;
							}
							if( cur[0] >= '0' && cur[0] <= '9' )
							{
								int val = hkString::atoi( cur );
	#							if defined(HK_DEBUG)
								if( val != 0 )
								{
									const char* extraWarning = "'.";
									for( int i = 0; i < cenum.getNumItems(); ++i )
									{
										if( cenum.getItem(i).getValue() & val )
										{
											extraWarning = "'. Some bits conflict with reflected bits.";
											break;
										}
									}
									HK_WARN(0x555b54ac, "Unreflected bits found in flags - using them anyway. '"
										<< cur << "' found for '" << cenum.getName() << "' in member '" << member->getName()
										<< extraWarning );
								}
	#							endif
								accum |= val;
							}
							else
							{
								int val = 0;
								if( cenum.getValueOfName( cur, &val ) == HK_SUCCESS )
								{
									accum |= val;
								}
								else
								{
									HK_WARN(0x555b54ab, "Invalid flag string '" << cur << "' found for '"
										<< cenum.getName() << "' in member '" << member->getName() );
								}
							}
							cur = next;
						}
						member->setEnumValue( buffer.pointerAt(classStartOffset + member->getOffset()), accum);
					}
					break;
				}
				default:
				{
					HK_ASSERT2(0x58b01b1f,0,"Found unknown (or unhandled) class member type in XML read.");
					break;
				}
			}
			if( consumeEndElement(parser, reader, startElement) == HK_FAILURE )
			{
				delete node;
				break;
			}
		}
		else if ( hkXmlParser::EndElement* ee = node->asEnd() )
		{
			if( topElement && (ee->name == topElement->name ))
			{
				delete node;
				return HK_SUCCESS;
			}
		}
		delete node;
	}
	return HK_FAILURE;
}

static int readEmbeddedHomogeneousClass( hkXmlParser& parser, hkStreamReader* reader, hkXmlParser::StartElement* startElement, Buffer& localBuffer, Buffer& buffer, hkRelocationInfo& reloc, hkStringMap<void*>* nameToObject )
{
	// Load the embedded class
	int classOffset = -1;
	{
		// Grab next node
		hkXmlParser::Node* snode = HK_NULL;
		parser.nextNode(&snode, reader);
		hkXmlParser::StartElement* classStart = snode->asStart();
		HK_ON_DEBUG(const char* klassName = classStart ? classStart->getAttribute("class", HK_NULL) : HK_NULL);
		HK_ASSERT2(0x0576de34, classStart && klassName && hkString::strCmp("hkClass", klassName) == 0, "Parse error, expected homogeneous class object.");

		// Load in data as hkClass
		hkRelocationInfo localReloc;
		localBuffer.reserve(hkClassClass.getObjectSize(), Buffer::PAD_16);
		localBuffer.advance(hkClassClass.getObjectSize(), Buffer::PAD_16);
		HK_ON_DEBUG(hkResult result = )readClassBody( hkClassClass, 0, localBuffer, classStart, parser, reader, localReloc, nameToObject );
		HK_ASSERT2(0x43e32345, result == HK_SUCCESS, "Unable to read embedded class for homogenous array");

		// Copy class to main buffer
		int localsize = localBuffer.reserve(0, Buffer::PAD_16);
		classOffset = buffer.reserve( localsize, Buffer::PAD_16 );
		buffer.advance( localsize, Buffer::PAD_16 );
		hkString::memCpy( buffer.pointerAt(classOffset), localBuffer.pointerAt(0), localsize );

		// Copy relocs
		{
			int i;
			for (i=0; i < localReloc.m_local.getSize() ;i++)
			{
				hkRelocationInfo::Local& local = localReloc.m_local[i];
				reloc.addLocal( local.m_fromOffset + classOffset, local.m_toOffset + classOffset );
			}

			for (i=0; i < localReloc.m_global.getSize() ;i++)
			{
				hkRelocationInfo::Global& global = localReloc.m_global[i];
				void * addr = global.m_toAddress;
				reloc.addGlobal( global.m_fromOffset + classOffset, addr, global.m_toClass );
			}

			for (i=0; i < localReloc.m_finish.getSize() ;i++)
			{
				hkRelocationInfo::Finish& finish = localReloc.m_finish[i];
				reloc.addFinish( finish.m_fromOffset + classOffset, finish.m_className );
			}
			for (i=0; i < localReloc.m_imports.getSize() ;i++)
			{
				hkRelocationInfo::Import& external = localReloc.m_imports[i];
				reloc.addImport( external.m_fromOffset + classOffset, external.m_identifier);
			}
		}

		// Apply fixups to local buffer
		localReloc.applyLocalAndGlobal( localBuffer.pointerAt(0) );

		delete snode;
	}
	return classOffset;
}

int hkXmlObjectReader::readObject(
	hkStreamReader* reader,
	void* buf, int bufSize,
	const hkClass& klass,
	hkRelocationInfo& reloc )
{
	hkArray<char> array;
	array.reserve(bufSize);
	if( readObject(reader, array, klass, reloc ) == HK_SUCCESS )
	{
		if( array.getSize() <= bufSize )
		{
			hkString::memCpy( buf, array.begin(), array.getSize() );
			return array.getSize();
		}
	}
	return -1;
}

hkResult hkXmlObjectReader::readObject(
	hkStreamReader* reader,
	hkArray<char>& array,
	const hkClass& klass,
	hkRelocationInfo& reloc )
{
	HK_ON_DEBUG( char peekTmp = 0 );
	HK_ASSERT2( 0x5412ce0d, reader->peek(&peekTmp,1) == 1, "Stream needs to support marking");
	hkXmlParser::Node* node;
	Buffer buffer(array);
	hkResult result = HK_FAILURE;

	while( m_parser->nextNode(&node, reader) == HK_SUCCESS )
	{
		if( hkXmlParser::StartElement* startElement = node->asStart() )
		{
			if( startElement->name == "hkobject")
			{
				int objectStart = buffer.reserve(klass.getObjectSize(), Buffer::PAD_16);
				buffer.advance(klass.getObjectSize(), Buffer::PAD_16);
				reloc.addFinish( objectStart, klass.getName() );
				result = readClassBody( klass, objectStart, buffer, startElement, *m_parser, reader, reloc, m_nameToObject );
			}
			else
			{
				HK_ASSERT3(0x5ae0b569, 0, "Unknown tag " << startElement->name );
			}
		}
		else if( hkXmlParser::Characters* characters = node->asCharacters() )
		{
			characters->canonicalize();
			HK_ASSERT3(0x742a0073, characters->text.getLength()==0, "unexpected characters" << startElement->name );

		}
		else if( hkXmlParser::EndElement* endElement = node->asEnd() )
		{
			HK_ASSERT3(0x46a5a10e, 0, "unexpected end node" << endElement->name );
		}
		else
		{
			HK_ERROR(0x6a858ec3, "Unknown element type returned from XML parser.");
		}
		delete node;
		break;
	}
	return result;
}

static int readChar(hkStreamReader* r)
{
	char c;
	if( r->read(&c, 1) == 1 )
	{
		return c;
	}
	return -1;
}

static void eatMarker( hkStreamReader* reader, const char* marker )
{
	const char* cur = marker;

	while(1)
	{
		int c = readChar(reader);
		HK_ASSERT( 0x5e1b58b1, c != -1 );
		if( !isSpace(c) )
		{
			HK_ASSERT2( 0x3d92ea88, c == *cur, "Eat marker broken in XML reader." );
			cur += 1;
			break;
		}
	}
	while( *cur )
	{
		HK_ON_DEBUG(int c = )readChar(reader);
		HK_ASSERT2( 0x2f41c29e, c != -1, "Eat marker broken in XML reader."  );
		HK_ASSERT2( 0x3ceebe22, c == *cur, "Eat marker broken in XML reader."  );
		cur += 1;
	}
}

hkResult hkXmlObjectReader::readRaw( hkStreamReader* reader, void* buf, int bufLen )
{
	eatMarker(reader, "<![CDATA[");
	hkResult result = base64read( reader, buf, bufLen );
	if( result == HK_SUCCESS )
	{
		eatMarker(reader, "]]>");
	}
	return result;
}

hkResult hkXmlObjectReader::base64read( hkStreamReader* sr, void* buf, int len )
{
	HK_ON_DEBUG( char peekTmp = 0 );
	HK_ASSERT2( 0x5412ce0d, sr->peek(&peekTmp,1) == 1, "Stream needs to support marking");

	//XXX batch this instead of single byte reads.
	static const signed char ascii2bin[128] =
	{
		-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
		-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
		-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
		52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
		-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
		15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
		-1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
		41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1
	};

	hkUint8* cur = static_cast<hkUint8*>(buf);
	const int inBytesPerOutByte[3] = { 0, 2, 3 };
	const int outBytesPerInBytes[4] = { 0, 0, 1, 2 };
	int bytesNeeded = (len/3)*4 + inBytesPerOutByte[len%3];

	int ibuflen = 0;
	unsigned char ibuf[4] = { 0 }; // ibuf contains only 6 bit chars
	while( bytesNeeded > 0 )
	{
		// read char by char discarding unknown chars
		unsigned char inchar;
		if( sr->read(&inchar, 1) != 1 )
		{
			return HK_FAILURE; // exit
		}
		if( ascii2bin[inchar & 0x7f] != -1 ) // to 6 bit
		{
			--bytesNeeded;
			ibuf[ ibuflen++ ] = hkUint8(ascii2bin[inchar]);

			if( ibuflen == 4 ) // got a chunk
			{
				ibuflen = 0;
				cur[0] = hkUint8((ibuf[0] << 2) | (ibuf[1] >> 4 ));
				cur[1] = hkUint8((ibuf[1] << 4) | (ibuf[2] >> 2 ));
				cur[2] = hkUint8((ibuf[2] << 6) | ibuf[3] );
				cur += 3;
				ibuf[0] = ibuf[1] = ibuf[2] = ibuf[3] = 0;
			}
		}
	}

	// eat padding '='
	{
		unsigned char c = 0;
		while( sr->peek( &c, 1 ) == 1 )
		{
			sr->skip(1);
			if( c != '=' )
			{
				break;
			}
		}
	}

	if( ibuflen ) // handle leftovers
	{
		unsigned char obuf[3];
		obuf[0] = hkUint8((ibuf[0] << 2) | (ibuf[1] >> 4 ));
		obuf[1] = hkUint8((ibuf[1] << 4) | (ibuf[2] >> 2 ));
		obuf[2] = hkUint8((ibuf[2] << 6) | ibuf[3] );

		for( int i = 0; i < outBytesPerInBytes[ibuflen]; ++i )
		{
			cur[i] = obuf[i];
		}
	}
	return HK_SUCCESS;
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
