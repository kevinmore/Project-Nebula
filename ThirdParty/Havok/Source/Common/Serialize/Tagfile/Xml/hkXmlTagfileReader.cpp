/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Tagfile/Xml/hkXmlTagfileReader.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Serialize/Data/hkDataObject.h>
#include <Common/Serialize/Util/Xml/hkXmlStreamParser.h>

namespace 
{

struct ReferenceInfo
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_SERIALIZE, ReferenceInfo);

	ReferenceInfo(): m_object(HK_NULL) {}

	void remember( hkDataObject::Value value)
	{
		m_objectRefs.pushBack(value);
	}
	void remember( hkDataArray::Value value)
	{
		m_arrayRefs.pushBack(value);
	}
	void assign()
	{
		HK_ASSERT(0x243423aa, m_object);
		for( int i = 0; i < m_arrayRefs.getSize(); ++i )
		{
			m_arrayRefs[i] = m_object;
		}
		for( int i = 0; i < m_objectRefs.getSize(); ++i )
		{
			m_objectRefs[i] = m_object;
		}
	}
		/// The object being referenced. HK_NULL if not defined yet.
	hkDataObjectImpl* m_object;
		/// The references to the object
	hkArray<hkDataArray::Value> m_arrayRefs;
	hkArray<hkDataObject::Value> m_objectRefs;
};
		
struct Reader
{
	typedef hkXmlStreamParser::Token Token;

	typedef hkDataObjectImpl ObjectImpl;
	typedef hkDataClassImpl ClassImpl;
	typedef hkPointerMap<int, ObjectImpl*> ReadObjects;
	typedef hkArray<ClassImpl*> ReadClasses;

		/// Ctor
	Reader(hkStreamReader* sr, hkDataWorld* cont);
		/// Dtor
	~Reader();
		
	hkResult parseRoot(hkDataObject& objOut);
	hkResult readHeader(hkXmlTagfile::Header& out);

protected:
	template <typename T>
	hkResult _parseObjectReferenceImpl(T& valueOut);

	void _pushBlock();
	hkResult _popBlock();

	const char* _getString(const hkSubString& text);
	const char* _getAttribute(const char* key);
	Token _skipWhiteSpace();
	hkBool _getBoolAttribute(const char* key, hkBool defaultValue);
	hkResult _parseClassDefinition();
	hkDataObject::Type _getType(const char* name, const char* className);
	//hkResult _parseObjectInstance(const hkDataClassImpl* klassImpl, hkDataObject& objOut);
	hkResult _parseMemberValue(  const hkDataClass::MemberInfo& minfo, hkDataObject::Value& valueOut );
	hkResult _parseInt(const char* blockName, hkInt64& valueOut);
	hkResult _parseReal(const char* blockName, hkReal& valueOut);
	hkResult _parseArrayItems(int numElems, hkDataArray& arr);
	hkResult _parseRealVector(hkReal* out, int numReal);
	hkResult _parseString(const char*& stringOut);
	hkResult _parseObjectReference(hkDataArray::Value& valueOut) { return _parseObjectReferenceImpl(valueOut); }
	hkResult _parseObjectReference(hkDataObject::Value& valueOut) { return _parseObjectReferenceImpl(valueOut); }
	hkResult _parseAndCreateObject(const hkDataClassImpl* klassImpl, hkDataObject& objOut);
	hkResult _parseObject(hkDataObject& obj);


	hkResult _parseIntArray(const char* blockName, hkDataArray& arr);
	hkResult _parseRealArray(const char* blockName, hkDataArray& arr);

		/// Given a key returns the reference info associated with that key
	ReferenceInfo* _findReferenceInfo(const char* key);
		/// Add the object with the reference
	hkResult _setReferencedObject(const char* id, hkDataObject& obj);
		/// Given the key either returns the reference info assoicated with that key, or creates and adds new reference info structure
	ReferenceInfo* _requireReferenceInfo(const char* key);

		/// Goes through all the references structures checking each as the object definition, and then 
		/// does the assignment 
	hkResult _fixUpReferences();


	hkXmlStreamParser m_parser;
	hkDataWorld* m_world;
	ReadClasses m_classes;
	hkArray<char*> m_prevStrings;
	int m_numPrevStringsStatic;

	// Maps reference 
	hkStorageStringMap<int> m_stringMap;
	hkArray<const char*> m_blockStack;

	// Maps the reference id to the information about the reference
	hkStringMap<ReferenceInfo*> m_referenceMap;

	hkArray<hkDataObject> m_tmpObjects;

	const char* m_null;
};

Reader::Reader(hkStreamReader* sr, hkDataWorld* cont)
	: m_parser(sr)
	, m_world(cont)
	, m_numPrevStringsStatic(0)
{
	m_classes.pushBack(HK_NULL);
	m_prevStrings.pushBack(const_cast<char*>("")); // 0
	m_prevStrings.pushBack(HK_NULL); // -1
	m_numPrevStringsStatic = 2;

	// A reference to #0000 is null
	m_null = _getString(hkSubString("#0000"));
}

Reader::~Reader()
{
	hkStringMap<ReferenceInfo*>::Iterator iter = m_referenceMap.getIterator();
	for (; m_referenceMap.isValid(iter); iter = m_referenceMap.getNext(iter))
	{
		ReferenceInfo* info = m_referenceMap.getValue(iter);
		delete info;
	}
}

template <typename T>
hkResult Reader::_parseObjectReferenceImpl(T& valueOut)
{
	_skipWhiteSpace();

	Token tok = m_parser.getToken();

	if (tok == hkXmlStreamParser::TOKEN_BLOCK_START && m_parser.getBlockName() == "ref")
	{
		_pushBlock();
		m_parser.advance();
		tok = _skipWhiteSpace();

		if (tok != hkXmlStreamParser::TOKEN_TEXT)
		{
			return HK_FAILURE;
		}

		// Get the id
		const char* id = _getString(m_parser.getLexeme());

		if (id == m_null)
		{
			valueOut.setNull();
		}
		else
		{
			ReferenceInfo* info = _requireReferenceInfo(id);
			// Remember the value 
			info->remember(valueOut);
		}
		
		// Read the rest
		m_parser.advance();
		_skipWhiteSpace();
		return _popBlock();
	}

	if (tok == hkXmlStreamParser::TOKEN_BLOCK_START_END && m_parser.getBlockName() == "null")
	{
		// Its null
		valueOut.setNull();
		return HK_SUCCESS;
	}

	return HK_FAILURE;
}

const char* Reader::_getString(const hkSubString& text)
{
	hkInplaceArray<char, 128> buffer;

	const int len = text.length();
	buffer.setSize(len + 1);

	hkString::strNcpy(buffer.begin(), text.m_start, len);
	buffer[len] = 0;

	return m_stringMap.insert(buffer.begin(), 1);
}

/* static */hkDataObject::Type Reader::_getType(const char* name, const char* className)
{
	hkTypeManager& typeManager = m_world->getTypeManager();

	switch (name[0])
	{
		case 'v':
		{
			if (hkString::strCmp(name, "void") == 0)
			{
				return typeManager.getSubType(hkTypeManager::SUB_TYPE_VOID);
			}
			if (hkString::strNcmp(name, "vec", 3) == 0)
			{
				const char* start = name + 3;
				const char* cur = start;
				while (*cur && *cur >= '0' && *cur <= '9') cur++;
				if (*cur != 0)
				{
					return HK_NULL;
				}
				const int size = hkString::atoi(start);
				return typeManager.makeTuple(typeManager.getSubType(hkTypeManager::SUB_TYPE_REAL), size);
			}
			return HK_NULL;
		}
		case 'b': 
		{
			if (hkString::strCmp(name, "byte") == 0)
			{
				return typeManager.getSubType(hkTypeManager::SUB_TYPE_BYTE);
			}
			break;
		}
		case 'i': 
		{
			if (hkString::strCmp(name, "int") == 0)
			{
				return typeManager.getSubType(hkTypeManager::SUB_TYPE_INT);
			}
			break;
		}
		case 'r': 
		{
			if (hkString::strCmp(name, "real") == 0)
			{
				return typeManager.getSubType(hkTypeManager::SUB_TYPE_REAL);
			}
			if (hkString::strCmp(name, "ref") == 0)
			{
				return typeManager.makePointer(typeManager.addClass(className));
			}
			break;
		}
		case 's':
		{
			if (hkString::strCmp(name, "string") == 0)
			{
				return typeManager.getSubType(hkTypeManager::SUB_TYPE_CSTRING);
			}
			if (hkString::strCmp(name, "struct") == 0)
			{
				return typeManager.addClass(className);
			}
			break;
		}
		default: break;
	}
	return HK_NULL;
}

void Reader::_pushBlock()
{
	const char* text = _getString(m_parser.getBlockName());
	m_blockStack.pushBack(text);
}

hkResult Reader::_popBlock()
{
	_skipWhiteSpace();

	HK_ASSERT(0x2442a2a3, m_parser.getToken() == hkXmlStreamParser::TOKEN_BLOCK_END);

	const char* text = _getString(m_parser.getBlockName());

	if (m_blockStack.getSize() <= 0)
	{
		return HK_FAILURE;
	}
	if (text != m_blockStack.back())
	{
		return HK_FAILURE;
	}

	// Next
	m_parser.advance();
	// 
	m_blockStack.popBack();
	return HK_SUCCESS;
}

hkResult Reader::_parseClassDefinition()
{
	HK_ASSERT(0x3424234, m_parser.getToken() == hkXmlStreamParser::TOKEN_BLOCK_START && m_parser.getBlockName() == "class");

	_pushBlock();

	const char* className = _getAttribute("name");
	if (!className)
	{
		return HK_FAILURE;
	}

	int version = 0;
	m_parser.getIntAttribute("version", version);

	const char* parentClassName = _getAttribute("parent");

	hkDataClass::Cinfo cinfo;

	cinfo.name = className;
	cinfo.version = static_cast<int>(version);
	cinfo.parent = parentClassName;

	hkTypeManager& typeManager = m_world->getTypeManager();
		
	// Read the members
	Token tok = m_parser.advance();
	while (true)
	{
		tok = _skipWhiteSpace();

		if (tok != hkXmlStreamParser::TOKEN_BLOCK_START_END || m_parser.getBlockName() != "member")
		{
			break;
		}
		// Add the member

		const char* memberName = _getAttribute("name");
		const char* memberTypeName = _getAttribute("type");
		const char* memberClassName = _getAttribute("class");

		if (memberName == HK_NULL || memberTypeName == HK_NULL)
		{
			HK_WARN(0x234243a2, "Member needs to have a name and a type");
			return HK_FAILURE;
		}
		
		hkDataObject::Type memberType = _getType(memberTypeName, memberClassName);
		if (memberType == HK_NULL)
		{
			HK_WARN(0x2342a3a4, "Unknown type '" << memberTypeName << "'");
			return HK_FAILURE;
		}

		if ( _getBoolAttribute("array", false) )
		{
			memberType = typeManager.makeArray(memberType);
		}

		int tupleCount = 0;
		m_parser.getIntAttribute("count", tupleCount);

		if (tupleCount > 0)
		{
			memberType = typeManager.makeTuple(memberType, tupleCount);
		}

		hkDataClass::Cinfo::Member& member = cinfo.members.expandOne();

		member.name = memberName;
		member.type = memberType;
		member.valuePtr = HK_NULL;						// ??
	
		if( memberClassName && m_world->findClass(memberClassName) == HK_NULL)
		{
			HK_ASSERT3(0x3f64fd57, m_world->getType() != hkDataWorld::TYPE_NATIVE,
				"The class " << className << " is not registered in the provided native data world.\n"\
				"Did you set class name and type name registries for the world?");
		}

		// Next token
		tok = m_parser.advance();
	}

	if ( _popBlock() == HK_FAILURE)
	{
		return HK_FAILURE;
	}
	
	// Next token
	m_parser.advance();
	// Add it
	m_world->newClass(cinfo);
	return HK_SUCCESS;
}

hkResult Reader::_parseInt(const char* blockName, hkInt64& valueOut)
{
	if (!(m_parser.getToken() == hkXmlStreamParser::TOKEN_BLOCK_START && m_parser.getBlockName() == blockName))
	{
		return HK_FAILURE;
	}
	_pushBlock();

	Token tok = m_parser.advance();
	tok = _skipWhiteSpace();

	if (tok != hkXmlStreamParser::TOKEN_TEXT)
	{
		return HK_FAILURE;
	}

	// Get an int
	hkResult res = hkXmlStreamParser::parseInt(m_parser.getLexeme(), valueOut);
	if (res != HK_SUCCESS)
	{
		return res;
	}
	
	m_parser.advance();
	return _popBlock();
}

hkResult Reader::_parseReal(const char* blockName, hkReal& valueOut)
{
	if (!(m_parser.getToken() == hkXmlStreamParser::TOKEN_BLOCK_START && m_parser.getBlockName() == blockName))
	{
		return HK_FAILURE;
	}
	_pushBlock();

	Token tok = m_parser.advance();
	tok = _skipWhiteSpace();

	if (tok != hkXmlStreamParser::TOKEN_TEXT)
	{
		return HK_FAILURE;
	}

	// Get an real
	hkResult res = hkXmlStreamParser::parseReal(m_parser.getLexeme(), valueOut);
	if (res != HK_SUCCESS)
	{
		return res;
	}
	m_parser.advance();
	return _popBlock();
}

/* 
static void _ExtractWhiteSpaceDelimited(const hkSubString& subString, hkArray<hkSubString>& out)
{
	const char* start = subString.m_start;
	const char* end = subString.m_end;

	while (start < end)
	{
		// Skip the white space
		while (start < end && hkXmlLexAnalyzer::isWhiteSpace(*start)) start++;
		// Now need to find the end

		const char* cur = start;
		while (cur < end && !hkXmlLexAnalyzer::isWhiteSpace(*cur)) cur++;

		if (cur - start > 0)
		{
			out.expandOne().set(start, cur);
		}
	
		// Next
		start = cur;
	}
} */

hkResult Reader::_parseRealVector(hkReal* out, int numReal)
{
	_pushBlock();
	Token tok = m_parser.advance();
	
	for (int i = 0; i < numReal; i++)
	{
		tok = _skipWhiteSpace();

		if (tok != hkXmlStreamParser::TOKEN_TEXT)
		{
			return HK_FAILURE;
		}

		hkSubString lexeme = m_parser.getLexeme();
		hkResult res = hkXmlStreamParser::parseReal(lexeme, out[i]);
		if (res != HK_SUCCESS)
		{
			return res;
		}

		// Next
		tok = m_parser.advance();
	}

	return _popBlock();
}

hkResult Reader::_parseString(const char*& stringOut)
{
	stringOut = HK_NULL;

	hkSubString blockName = m_parser.getBlockName();
	if (blockName == "null" && hkXmlStreamParser::TOKEN_BLOCK_START_END)
	{
		m_parser.advance();
		return HK_SUCCESS;
	}

	if (blockName != "string")
	{
		return HK_FAILURE;
	}
	if (m_parser.getToken() == hkXmlStreamParser::TOKEN_BLOCK_START_END)
	{
		m_parser.advance();
		// Its an empty block
		stringOut = "";
		return HK_SUCCESS;
	}

	if (m_parser.getToken() != hkXmlStreamParser::TOKEN_BLOCK_START)
	{
		return HK_FAILURE;
	}
	_pushBlock();
	Token tok = m_parser.advance();

	// Check it its empty
	if (tok == hkXmlStreamParser::TOKEN_BLOCK_END)
	{
		stringOut = "";
		return _popBlock();
	}

	hkStringBuf buf;
	while (tok == hkXmlStreamParser::TOKEN_TEXT || tok == hkXmlStreamParser::TOKEN_WHITESPACE)
	{
		hkSubString lexeme = m_parser.getLexeme();
		// Append
		buf.append(lexeme.m_start, lexeme.length());
		// Next
		tok = m_parser.advance();
	}

	if (hkXmlStreamParser::needsDecode(hkSubString(buf.cString(), buf.getLength())))
	{
		hkStringBuf decodeBuf;
		hkXmlStreamParser::decodeString(hkSubString(buf.cString(), buf.getLength()), decodeBuf);
		buf = decodeBuf;
	}
	
	// Put the string in the buffer
	stringOut = m_stringMap.insert(buf.cString(), 1);

	return _popBlock();
}

hkResult Reader::_parseIntArray(const char* blockName, hkDataArray& arr)
{
	const int size = arr.getSize();

	for (int i = 0; i < size; i++ )
	{
		_skipWhiteSpace();

		if (m_parser.getToken() == hkXmlStreamParser::TOKEN_BLOCK_START)
		{
			hkInt64 v;
			hkResult res = _parseInt(blockName, v);
			if (res != HK_SUCCESS)
			{
				return res;
			}
			arr[i] = v;
			continue;
		}

		
		if (m_parser.getToken() == hkXmlStreamParser::TOKEN_TEXT)
		{
			hkSubString lexeme = m_parser.getLexeme();
			hkInt64 v;
			hkResult res = hkXmlStreamParser::parseInt(lexeme, v);
			if (res != HK_SUCCESS)
			{
				return res;
			}
			arr[i] = v;

			m_parser.advance();
			continue;
		}

		return HK_FAILURE;
	}

	return HK_SUCCESS;
}

hkResult Reader::_parseRealArray(const char* blockName, hkDataArray& arr)
{
	const int size = arr.getSize();

	for (int i = 0; i < size; i++ )
	{
		_skipWhiteSpace();

		if (m_parser.getToken() == hkXmlStreamParser::TOKEN_BLOCK_START)
		{
			hkReal v;
			hkResult res = _parseReal(blockName, v);
			if (res != HK_SUCCESS)
			{
				return res;
			}
			arr[i] = v;
			continue;
		}

		if (m_parser.getToken() == hkXmlStreamParser::TOKEN_TEXT)
		{
			hkReal v;
			hkResult res = hkXmlStreamParser::parseReal(m_parser.getLexeme(), v);
			if (res != HK_SUCCESS)
			{
				return res;
			}
			arr[i] = v;
			m_parser.advance();
			continue;
		}

		return HK_FAILURE;
	}
	
	return HK_SUCCESS;
}

hkResult Reader::_parseArrayItems(int numElems, hkDataArray& arr)
{
	hkDataObject::Type type = arr.getType();
	switch( type->getSubType())
	{
		case hkTypeManager::SUB_TYPE_BYTE:
		case hkTypeManager::SUB_TYPE_INT:
		{
			const char* blockName = type->isByte() ? "byte" : "int";
			return _parseIntArray(blockName, arr);
		}
		case hkTypeManager::SUB_TYPE_REAL:
		{
			return _parseRealArray("real", arr);
		}
		case hkTypeManager::SUB_TYPE_TUPLE:
		{
			if (type->isVec())
			{
				const int numReals = type->getTupleSize();
				hkInplaceArray<hkReal, 16> r;
				r.setSize(numReals);

				for (int i = 0; i < numElems; i++)
				{
					_skipWhiteSpace();
					if (_parseRealVector(r.begin(), numReals) != HK_SUCCESS)
					{
						return HK_FAILURE;
					}
					arr[i].setVec(r.begin(), numReals);
				}
				break;
			}

			HK_ASSERT(0x234324a, !"Other tuple types are not handled");

			break;
		}
		case hkTypeManager::SUB_TYPE_CSTRING:
		{
			for( int i = 0; i < numElems; ++i )
			{
				_skipWhiteSpace();
				const char* s;
				if (_parseString(s) != HK_SUCCESS)
				{
					return HK_FAILURE;
				}
				arr[i] = s;
			}
			break;
		}
		case hkTypeManager::SUB_TYPE_CLASS:
		{			
			// Must be a reference or NULL there are no other choices
			for( int i = 0; i < numElems; ++i )
			{
				_skipWhiteSpace();
				//Token tok = m_parser.getToken();

				{
					hkDataObject obj = arr[i].asObject();

					// Keep access thru this object in scope
					m_tmpObjects.pushBack(obj);

					hkResult res = _parseObject(obj);
					if (res != HK_SUCCESS)
					{
						return res;
					}
				}
			}
			break;
		}
		case hkTypeManager::SUB_TYPE_POINTER:
		{
			// Must be a reference or NULL there are no other choices
			for( int i = 0; i < numElems; ++i )
			{
				hkDataArray::Value value = arr[i];
				_parseObjectReference(value);
			}
			break;
		}
		default:
		{
			HK_ASSERT(0x618f9194, 0);
			return HK_FAILURE;
		}
	}

	_skipWhiteSpace();
	return HK_SUCCESS;
}

hkResult Reader::_parseMemberValue(  const hkDataClass::MemberInfo& minfo, hkDataObject::Value& valueOut )
{
	hkDataObject::Type type = minfo.m_type;

	if( type->isTuple())
	{
		hkResult res;
		int size = type->getTupleSize();

		if (type->isVec())
		{
			// It's really a vec type
			hkReal r[16];
			HK_ASSERT(0x42424bb2, size <= (int) HK_COUNT_OF(r));
			if (_parseRealVector(r, size) != HK_SUCCESS)
			{
				return HK_FAILURE;
			}
			valueOut.setVec(r, size);
			return HK_SUCCESS;
		}

		if (m_parser.hasAttribute("size"))
		{
			res = m_parser.getIntAttribute("size", size);
			if (res != HK_SUCCESS)
			{
				return res;
			}

			if (size != type->getTupleSize())
			{
				HK_WARN(0x43243a2a, "Tuple count of member is different from the data definition");
				return HK_FAILURE;
			}
		}

		hkDataObject obj(valueOut.m_impl);
		hkDataArray arr = obj[minfo.m_name].asArray();

		HK_ASSERT(0x242343aa, arr.getSize() == size);

		if (m_parser.getToken() != hkXmlStreamParser::TOKEN_BLOCK_START || m_parser.getBlockName() != "tuple")
		{
			HK_WARN(0x2342a34a, "Expecting <tuple> block start");
			return HK_FAILURE;
		}

		_pushBlock();
		m_parser.advance();

		res = _parseArrayItems( type->getTupleSize(), arr  );
		if (res != HK_SUCCESS)
		{
			return res;
		}

		return _popBlock();
	}
	else if( type->isArray())
	{
		const hkDataClass::MemberInfo& arrayInfo = minfo;

		int size;
		hkResult res = m_parser.getIntAttribute("size", size);
		if (res != HK_SUCCESS)
		{
			return res;
		}

		hkDataObject obj(valueOut.m_impl);
		hkDataObject::Value value = obj[arrayInfo.m_name];
		hkDataArray arr( m_world->newArray(obj, value.m_handle, arrayInfo) );

		arr.setSize(size);

		_pushBlock();
		m_parser.advance();
		if (_parseArrayItems( size, arr) != HK_SUCCESS)
		{
			return HK_FAILURE;
		}
		return _popBlock();
	}

	switch( type->getSubType())
	{
		case hkTypeManager::SUB_TYPE_BYTE:
		{
			hkInt64 i;
			hkResult res = _parseInt("byte", i);
			if (res != HK_SUCCESS)
			{
				return res;
			}
			valueOut = i;
			break;
		}
		case hkTypeManager::SUB_TYPE_INT:
		{
			hkInt64 i;
			hkResult res = _parseInt("int", i);
			if (res != HK_SUCCESS)
			{
				return res;
			}
			valueOut = i;
			break;
		}
		case hkTypeManager::SUB_TYPE_REAL:
		{
			hkReal v;
			hkResult res = _parseReal("real", v);
			if (res != HK_SUCCESS)
			{
				return res;
			}
			valueOut = v;
			break;
		}
		case hkTypeManager::SUB_TYPE_CLASS:
		{
			hkDataClassImpl* cls = m_world->findClass(type->getTypeName());
			if (!cls)
			{
				HK_ASSERT(0x234342a, !"Couldn't find class");
				return HK_FAILURE;
			}

			hkDataObject obj;
			hkResult res = _parseAndCreateObject(cls, obj);
			if (res != HK_SUCCESS)
			{
				return res;
			}
			valueOut = obj;
			break;
		}
		case hkTypeManager::SUB_TYPE_POINTER:
		{
			// Can only be a reference
			return _parseObjectReference(valueOut);
		}
		case hkTypeManager::SUB_TYPE_CSTRING:
		{
			const char* s;
			hkResult res = _parseString(s);
			if (res != HK_SUCCESS)
			{
				return HK_FAILURE;
			}
			valueOut = s;
			break;
		}
		default:
		{
			HK_ASSERT(0x4c3c5273, 0);
			return HK_FAILURE;
		}
	}

	return HK_SUCCESS;
}

const char* Reader::_getAttribute(const char* key)
{
	hkSubString value;
	if (m_parser.getValue(key, value) != HK_SUCCESS)
	{
		return HK_NULL;
	}

	if (value.length() >= 2 && value.m_start[0] == '"' && value.m_end[-1] == '"')
	{
		value.m_start++;
		value.m_end--;
		return _getString(value);
	}

	HK_ASSERT(0x3214a234, !"Badly formed attribute value (no quotes)");
	return HK_NULL;
}

hkBool Reader::_getBoolAttribute(const char* key, hkBool defaultValue)
{
	hkSubString value;
	if (m_parser.getValue(key, value) != HK_SUCCESS)
	{
		return defaultValue;
	}

	if (value == "\"true\"")
	{
		return true;
	}
	if (value == "\"false\"")
	{
		return false;
	}

	HK_WARN(0x24234aa3, "Invalid bool value " << key << "='" << value << "'");
	return defaultValue;
}

hkXmlStreamParser::Token Reader::_skipWhiteSpace()
{
	Token tok = m_parser.getToken();
	while (tok == hkXmlStreamParser::TOKEN_WHITESPACE)
	{
		tok = m_parser.advance();
	}
	return tok;
}

ReferenceInfo* Reader::_findReferenceInfo(const char* key)
{
	return m_referenceMap.getWithDefault(key, HK_NULL);
}

ReferenceInfo* Reader::_requireReferenceInfo(const char* key)
{
	ReferenceInfo* info = m_referenceMap.getWithDefault(key, HK_NULL);
	if (info)
	{
		return info;
	}

	info = new ReferenceInfo;
	m_referenceMap.insert(key, info);
	return info;
}

hkResult Reader::_setReferencedObject(const char* id, hkDataObject& obj)
{
	ReferenceInfo* info = _requireReferenceInfo(id);
	if (info->m_object)
	{
		HK_WARN(0x32423432, "The id '" << id << "' has already appeared");
		return HK_FAILURE;
	}

	info->m_object = obj.getImplementation();
	return HK_SUCCESS;
}

hkResult Reader::_parseAndCreateObject(const hkDataClassImpl* klassImpl, hkDataObject& objOut)
{
	HK_ASSERT(0x53454a35, m_parser.getToken() == hkXmlStreamParser::TOKEN_BLOCK_START && (m_parser.getBlockName() == "object" || m_parser.getBlockName() == "struct"));

	if (klassImpl == HK_NULL)
	{
		const char* typeName = _getAttribute("type");
		
		if (!typeName)
		{
			return HK_FAILURE;
		}

		klassImpl = m_world->findClass(typeName);
		if (!klassImpl)
		{
			HK_WARN(0x3243243, "Couldn't find the class definition for '" << typeName << "'");
			return HK_FAILURE;
		}
	}

	const char* id = _getAttribute("id");

	hkDataClass klass(const_cast<hkDataClassImpl*>(klassImpl));
	hkDataObject obj = m_world->newObject(klass);

	hkResult res = _parseObject(obj);
	if (res != HK_SUCCESS)
	{
		return res;
	}

	if (id)
	{
		res = _setReferencedObject(id, obj);
		if (res != HK_SUCCESS)
		{
			return res;
		}
	}

	objOut = obj;
	return HK_SUCCESS;
}

hkResult Reader::_parseObject(hkDataObject& obj)
{
	HK_ASSERT(0x53454a35, m_parser.getToken() == hkXmlStreamParser::TOKEN_BLOCK_START && (m_parser.getBlockName() == "object" || m_parser.getBlockName() == "struct"));

	// Push the block
	_pushBlock();

	hkDataClass klass(const_cast<hkDataClassImpl*>(obj.getImplementation()->getClass()));

	Token tok = m_parser.advance();
	while (true)
	{
		tok = _skipWhiteSpace();

		if (tok != hkXmlStreamParser::TOKEN_BLOCK_START)
		{
			break;
		}

		// Its a start of a block
		const char* memberName = _getAttribute("name");
		if (!memberName)
		{
			HK_WARN(0x3424a324, "Expecting members blocks to have associated 'name' fields");
			return HK_FAILURE;
		}

		int memberIndex = klass.getMemberIndexByName(memberName);
		if (memberIndex < 0)
		{
			HK_WARN(0x2343232, "The member '" << memberName <<"' is not defined in the class definition");
			return HK_FAILURE;
		}

		// Look up the member
		hkDataClass::MemberInfo memberInfo;
		klass.getMemberInfo(memberIndex, memberInfo);

		hkDataObject::Value value = obj[memberInfo.m_name];
		hkResult res = _parseMemberValue( memberInfo, value);
		if (res != HK_SUCCESS)
		{
			return res;
		}
	}

	if (_popBlock() != HK_SUCCESS)
	{
		return HK_FAILURE;
	}
	return HK_SUCCESS;
}

hkResult Reader::_fixUpReferences()
{
	hkStringMap<ReferenceInfo*>::Iterator iter = m_referenceMap.getIterator();
	for (; m_referenceMap.isValid(iter); iter = m_referenceMap.getNext(iter))
	{
		ReferenceInfo* info = m_referenceMap.getValue(iter);

		if (info->m_object == HK_NULL)
		{
			// 
			HK_WARN(0x34234a34, "Definition of object '" << m_referenceMap.getKey(iter) << "' was not found");
			return HK_FAILURE;
		}

		// Do the assignment
		info->assign();
	}

	return HK_SUCCESS;
}

hkResult Reader::readHeader(hkXmlTagfile::Header& out)
{
	Token tok = m_parser.advance();

	// Skip any white space
	tok = _skipWhiteSpace();
	// If its header
	if (tok == hkXmlStreamParser::TOKEN_QBLOCK)
	{
		tok = m_parser.advance();
	}

	// Skip any whitespace
	tok = _skipWhiteSpace();

	if (tok != hkXmlStreamParser::TOKEN_BLOCK_START)
	{
		return HK_FAILURE;
	}

	if (m_parser.getBlockName() != "hktagfile")
	{
		return HK_FAILURE;
	}

	if(m_parser.getIntAttribute("version", out.m_version) == HK_FAILURE)
	{
		return HK_FAILURE;
	}

	switch(out.m_version)
	{
		case 1:
		case 2: // version 2 introduces predicates
			{
				hkSubString sdkVersionStr;
				if (m_parser.getValue("sdkversion", sdkVersionStr) == HK_SUCCESS)
				{
					out.m_sdkVersion.set(sdkVersionStr.m_start+1, sdkVersionStr.length()-2);
				}

				break;
			}

		default:
			{
				HK_WARN_ALWAYS(0x23c6037f, "Unrecognised tagfile version " << out.m_version);
				return HK_FAILURE;
			}
	}

	return HK_SUCCESS;
}

hkResult Reader::parseRoot(hkDataObject& objOut)
{
	Token tok = m_parser.advance();

	// Skip any white space
	tok = _skipWhiteSpace();
	// If its header
	if (tok == hkXmlStreamParser::TOKEN_QBLOCK)
	{
		tok = m_parser.advance();
	}
	
	// Skip any whitespace
	tok = _skipWhiteSpace();

	if (tok != hkXmlStreamParser::TOKEN_BLOCK_START)
	{
		HK_WARN_ALWAYS(0xfeed00aa, "Didn't find the root 'hktagfile' block");
		return HK_FAILURE;
	}

	if (m_parser.getBlockName() != "hktagfile")
	{
		HK_WARN_ALWAYS(0xfeed00aa, "Expecting 'hktagfile' block");
		return HK_FAILURE;
	}

	int version = 0;
	m_parser.getIntAttribute("version", version);

	_pushBlock();

	tok = m_parser.advance();
	while (true)
	{
		tok = _skipWhiteSpace();

		if (tok != hkXmlStreamParser::TOKEN_BLOCK_START || m_parser.getBlockName() != "class")
		{
			break;
		}

		// We have a class definition
		_parseClassDefinition();
	}
	
	hkDataObject obj(HK_NULL);
	hkBool firstObj = true;

	while (true)
	{
		// Skip white space
		tok = _skipWhiteSpace();

		// 
		if (tok != hkXmlStreamParser::TOKEN_BLOCK_START && tok != hkXmlStreamParser::TOKEN_BLOCK_START_END)
		{
			break;
		}

		hkDataObject curObj;
		hkResult res = _parseAndCreateObject(HK_NULL, curObj);
		if (res == HK_FAILURE)
		{
			// Couldn't read it
			return HK_FAILURE;
		}

		if (firstObj)
		{
			obj = curObj;
			firstObj = true;
		}
	}

	if (_popBlock() != HK_SUCCESS)
	{
		HK_WARN(0x32423432, "Expecting block end");
		return HK_FAILURE;
	}

	tok = _skipWhiteSpace();
	if (tok != hkXmlStreamParser::TOKEN_EOF)
	{
		return HK_FAILURE;
	}

	{
		hkResult res = _fixUpReferences();
		if (res != HK_SUCCESS)
		{
			return res;
		}
	}

	objOut = obj;
	// Return the object
	return HK_SUCCESS;
}
	
} // anonymous

hkDataObject hkXmlTagfileReader::load( hkStreamReader* stream, hkDataWorld& world )
{
	Reader reader(stream, &world);

	hkDataObject obj;
	hkResult res = reader.parseRoot(obj);
	if (res != HK_SUCCESS)
	{
		HK_WARN(0x3244aa23, "Error parsing xml tagfile");
		return hkDataObject(HK_NULL);
	}
	return obj;
}

hkResult hkXmlTagfileReader::readHeader(hkStreamReader* stream, hkXmlTagfile::Header& out)
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
