/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>

#include <Common/Serialize/TypeManager/hkTypeManager.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Container/SubString/hkSubString.h>

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                       hkTypeManager::Type

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

int hkTypeManager::Type::getTupleSize() const
{
	return (m_subType == SUB_TYPE_TUPLE) ? m_extra.m_size : -1;
}

const char* hkTypeManager::Type::getTypeName() const
{
	return (m_subType == SUB_TYPE_CLASS) ? m_extra.m_name : HK_NULL;
}

/* static */hkUint32 hkTypeManager::Type::_calcHash(const Type* cur)
{
	hkUint32 hash = 0;

	do 
	{
		// Rotate
		hash = (hash << 1) | (hash >> 31);
		// Mix in the sub type
		hash ^= hkUint32(cur->m_subType) * hkUint32(2654435761u);
			
		switch (cur->m_subType)
		{
			case SUB_TYPE_CLASS:
			{
				// The type pointer makes it unique
				hash ^= hkUint32(hk_size_t(cur));
				break;
			}
			case SUB_TYPE_TUPLE:
			{
				hash ^= hkUint32(cur->m_extra.m_size);
				break;
			}
			default: break;
		}

		cur = cur->m_parent;
	}
	while (cur);

	return hash;
}

hkUint32 hkTypeManager::Type::calcHash() const
{
	const hkUint32 hash = _calcHash(this);
	// Make sure hash is never the hash 'empty' value
	return (hash == hkUint32(0xffffffff)) ? hkUint32(0x7af1f32a) : hash;
}

hkBool hkTypeManager::Type::_equals(const Type& rhs) const
{
	if (m_subType != rhs.m_subType || 
		m_parent != rhs.m_parent)
	{
		return false;
	}

	switch (m_subType)
	{
		case SUB_TYPE_CLASS:		return m_extra.m_name == rhs.m_extra.m_name;
		case SUB_TYPE_TUPLE:		return m_extra.m_size == rhs.m_extra.m_size;
		default:					return true;
	}
}

hkTypeManager::Type* hkTypeManager::Type::findTerminal()
{
	Type* cur = this;
	while (cur->m_parent)
	{
		cur = cur->m_parent;
	}
	return cur;
}

/* static */void HK_CALL hkTypeManager::Type::asText(const Type* cur, hkOstream& stream)
{
	while (cur)
	{
		switch (cur->m_subType)
		{
			case SUB_TYPE_INVALID:	stream << "!"; break;
			case SUB_TYPE_VOID:		stream << "void"; break;
			case SUB_TYPE_BYTE:		stream << "byte"; break;
			case SUB_TYPE_REAL:		stream << "real"; break;
			case SUB_TYPE_INT:		stream << "int"; break;
			case SUB_TYPE_CSTRING:	stream << "cstring"; break;
			case SUB_TYPE_CLASS:		
			{
				if (cur->getTypeName())
				{
					stream << "class " << cur->getTypeName();
				}
				else
				{
					stream << "homogeneous/variant class";
				}
				break;
			}
			case SUB_TYPE_POINTER:	stream << "*"; break;
			case SUB_TYPE_ARRAY:	stream << "[]"; break;
			case SUB_TYPE_TUPLE:	stream << "{" << cur->m_extra.m_size << "}"; break;
			case SUB_TYPE_COUNT_OF: break;
		}

		cur = cur->m_parent;
	}
}

void hkTypeManager::Type::asText(hkOstream& stream) const
{
	asText(this, stream);
}

hkStringPtr  hkTypeManager::Type::asString() const
{
	hkArray<char> buffer;
	hkOstream stream(buffer);

	asText(stream);

	buffer.pushBack(0);

	return buffer.begin();
}

/* static */void HK_CALL hkTypeManager::Type::getTypePath(Type* type, hkArray<Type*>& types)
{
	types.clear();
	while (type)
	{
		types.pushBack(type);
		type = type->getParent();
	}
}

/* static */hkBool HK_CALL hkTypeManager::Type::_isEqual(const Type* a, const Type* b)
{
	while (a && b)
	{
		if (a->m_subType != b->m_subType)
		{
			return false;
		}
		switch (a->m_subType)
		{
			case SUB_TYPE_TUPLE:
			{
				if (a->getTupleSize() != b->getTupleSize())
				{
					return false;
				}
				break;
			}
			case SUB_TYPE_CLASS:
			{
				
				// For the moment if either type name is NULL, the the class is said to match.

				const char* na = a->getTypeName();
				const char* nb = b->getTypeName();

				if (na == HK_NULL || nb == HK_NULL)
				{
					return true;
				} 
				else if (hkString::strCmp(na, nb) != 0)
				{
					return false;
				}
				break;
			}
			default: break;
		}
		
		a = a->getParent();
		b = b->getParent();
	}

	return a == HK_NULL && b == HK_NULL;
}

hkBool hkTypeManager::Type::isEqual(const Type* type) const
{
	if (type == this)
	{
		return true;
	}

	return _isEqual(this, type);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                           hkTypeManager

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

hkTypeManager::hkTypeManager():
	m_typeFreeList(sizeof(Type), HK_ALIGN_OF(Type), 2048)
{
	hkString::memSet(m_builtInTypes, 0, sizeof(m_builtInTypes));
	// Set up the defaults
	_addBuiltIn(SUB_TYPE_VOID);
	_addBuiltIn(SUB_TYPE_INT);
	_addBuiltIn(SUB_TYPE_REAL);
	_addBuiltIn(SUB_TYPE_CSTRING);
	_addBuiltIn(SUB_TYPE_BYTE);

	{
		// Get the type
		Type* type = (Type*)m_typeFreeList.alloc();
		type->m_subType = SUB_TYPE_CLASS;
		type->m_parent = HK_NULL;
		type->m_extra.m_name = HK_NULL;
	
		// Add the type
		m_typeMultiMap.insert(type->calcHash(), type);

		m_homogenousClass = type;
	}
}

hkTypeManager::~hkTypeManager()
{
}

hkTypeManager::Type* hkTypeManager::_addBuiltIn(SubType subType) 
{
	Type type;
	type.m_subType = subType;
	type.m_parent = HK_NULL;
	type.m_extra.m_name = HK_NULL;

	Type* newType = _addType(type);

	m_builtInTypes[subType] = newType;
	return newType;
}

hkTypeManager::Type* hkTypeManager::_addType(const Type& typeIn) 
{
	// Can't handle classes
	HK_ASSERT(0x3424aa23, typeIn.m_subType != SUB_TYPE_CLASS);
	const hkUint32 hash = typeIn.calcHash();
	
	hkPointerMultiMap<hkUint32, Type*>::Iterator iter = m_typeMultiMap.findKey(hash);
	for (; m_typeMultiMap.isValid(iter); iter = m_typeMultiMap.getNext(iter, hash))
	{
		Type* type = m_typeMultiMap.getValue(iter);
		if (typeIn._equals(*type))
		{
			return type;
		}
	}

	// Get the type
	Type* type = (Type*)m_typeFreeList.alloc();
	*type = typeIn;

	// Add the type
	m_typeMultiMap.insert(hash, type);

	return type;
}

void hkTypeManager::findTypesUsingClass(Type* clsType, hkArray<Type*>& types)
{
	HK_ASSERT(0x2343a432, clsType->isClass());

	types.clear();

	hkPointerMultiMap<hkUint32, Type*>::Iterator iter = m_typeMultiMap.getIterator();
	for (; m_typeMultiMap.isValid(iter); iter = m_typeMultiMap.getNext(iter))
	{
		Type* type = m_typeMultiMap.getValue(iter);

		if (type->findTerminal() == clsType)
		{
			types.pushBack(type);
		}
	}
}

void hkTypeManager::renameClass(const char* clsName, const char* newName)
{
	HK_ASSERT(0x4234324, _isValidClassName((newName)));

	Type* clsType = getClass(clsName);
	HK_ASSERT(0x324a234a, clsType);
	if (!clsType)
	{
		return;
	}

	if (getClass(newName))
	{
		HK_ASSERT(0x1232131, !"Can only rename to a new unique name");
		return;
	}

	// I can just rename in the type, because the hash will remain the same
	m_classMap.remove(clsName);
	clsType->m_extra.m_name = m_classMap.insert(newName, clsType);
}

void hkTypeManager::removeClass(Type* clsType)
{
	HK_ASSERT(0x2343432, clsType->isClass());

	// Get the name
	const char* clsName = clsType->getTypeName();

	hkArray<Type*> types;
	findTypesUsingClass(clsType, types);	
	// Have to pre calc the hashes, because as I remove objects the hashing could break
	// as parents are destroyed.
	hkArray<hkUint32> hashes;
	hashes.setSize(types.getSize());
	for (int i = 0; i < types.getSize(); i++)
	{
		hashes[i] = types[i]->calcHash();
	}

	for (int i = 0; i < types.getSize(); i++)
	{
		Type* type = types[i];

		// Remove from the multi map
		HK_ON_DEBUG(hkResult res = )m_typeMultiMap.remove(hashes[i], type);
		HK_ASSERT(0x2432423a, res == HK_SUCCESS);

		// Invalidate
		type->m_subType = SUB_TYPE_INVALID;
		type->m_parent = HK_NULL;
	}

	// Remove from the class map
	m_classMap.remove(clsName);
}

static void HK_CALL _freeListCallback(void* start, hk_size_t size, bool allocated,int pool,void* param)
{
	hkArray<hkTypeManager::Type*>& array = *(hkArray<hkTypeManager::Type*>*)param;
	if (allocated)
	{
		hkTypeManager::Type* type = (hkTypeManager::Type*)start;
		if (!type->isValid())
		{
			array.pushBack(type);
		}
	}
}

void hkTypeManager::garbageCollect()
{
	hkArray<Type*> array;
	m_typeFreeList.walkMemory(_freeListCallback, 0, &array);

	for (int i = 0; i < array.getSize(); i++ )
	{
		m_typeFreeList.free(array[i]);
	}

	m_typeFreeList.garbageCollect();
}

hkTypeManager::Type* hkTypeManager::makePointer(Type* parent)
{
	HK_ASSERT(0x23432423, parent && parent->isValid());

	Type type;
	type.m_subType = SUB_TYPE_POINTER;
	type.m_parent = parent;
	type.m_extra.m_name = HK_NULL;

	return _addType(type);
}

hkTypeManager::Type* hkTypeManager::makeArray(Type* parent)
{
	HK_ASSERT(0x23432423, parent && parent->isValid());

	Type type;
	type.m_subType = SUB_TYPE_ARRAY;
	type.m_parent = parent;

	return _addType(type);
}

hkTypeManager::Type* hkTypeManager::makeTuple(Type* parent, int tupleSize)
{
	HK_ASSERT(0x23423aa4, tupleSize > 0);
	HK_ASSERT(0x23432423, parent && parent->isValid());

	Type type;
	type.m_subType = SUB_TYPE_TUPLE;
	type.m_parent = parent;
	type.m_extra.m_size = tupleSize;

	return _addType(type);
}

/* static */hkBool HK_CALL hkTypeManager::_isValidClassName(const char* name)
{
	if (name == HK_NULL || name[0] == 0)
	{
		return false;
	}

	// First must be alpha or _
	{
		const char c = *name;

		if (!((c >= 'a' && c <= 'z') ||
			(c >= 'A' && c <= 'Z') ||
			(c == '_')))
		{
			return false;
		}
		name++;
	}

	// Can be alpha numeric, _ or :
	for (; *name;  name++)
	{
		const char c = *name;

		if ((c >= 'a' && c <= 'z') ||
			(c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') ||
			c == ':' ||
			c == '_')
		{
			continue;
		}

		return false;
	}
	return true;
}

hkTypeManager::Type* hkTypeManager::getClass(const char* clsName) const
{
	return m_classMap.getWithDefault(clsName, HK_NULL);
}

hkTypeManager::Type* hkTypeManager::addClass(const char* clsName)
{
	HK_ASSERT(0x23432a23, _isValidClassName(clsName));

	Type* prevType = getClass(clsName);
	if (prevType)
	{
		return prevType;
	}

	// Get the type
	Type* type = (Type*)m_typeFreeList.alloc();
	type->m_subType = SUB_TYPE_CLASS;
	type->m_parent = HK_NULL;
	type->m_extra.m_name = m_classMap.insert(clsName, type);

	// Add it to the types
	m_typeMultiMap.insert(type->calcHash(), type);

	// Return the type
	return type;
}

hkTypeManager::Type* hkTypeManager::replaceParent(Type* typeIn, Type* parent)
{
	if (typeIn->getParent() == parent)
	{
		return typeIn;
	}

	switch (typeIn->getSubType())
	{
		case SUB_TYPE_POINTER:	return makePointer(parent);
		case SUB_TYPE_ARRAY:	return makeArray(parent);
		case SUB_TYPE_TUPLE:	return makeTuple(parent, typeIn->getTupleSize());
		default:
		{
			HK_ASSERT(0x2344234, !"Cannot replace with a terminal type");
			return HK_NULL;
		}
	}
}

hkTypeManager::Type* hkTypeManager::replaceTerminal(Type* typeIn, Type* newTerminal)
{
	hkInplaceArray<Type*, 16> types;
	{
		Type* type = typeIn;
		do 
		{
			types.pushBack(type);
			type = type->m_parent;
		}
		while (type);
	}

	Type* oldTerminal = types.back();
	if (oldTerminal == newTerminal)
	{
		return typeIn;
	}

	Type* type = newTerminal;
	// Okay, we need to rebuild the type from the back
	for (int i = types.getSize() - 1; i >= 0; i--)
	{
		type = replaceParent(types[i], type);
	}
	return type;
}

hkTypeManager::Type* hkTypeManager::makeArray(SubType subType)
{
	Type* type = getSubType(subType);
	if (type == HK_NULL)
	{
		return HK_NULL;
	}
	return makeArray(type);
}

hkTypeManager::Type* hkTypeManager::getClassPointer(const char* name)
{
	return makePointer(addClass(name));
}

hkBool hkTypeManager::isOwned(Type* type) const
{
	// Look it up
	HK_ASSERT(0x32423423, type->isValid());

	const hkUint32 hash = type->calcHash();

	hkPointerMultiMap<hkUint32, Type*>::Iterator iter = m_typeMultiMap.findKey(hash);
	for (; m_typeMultiMap.isValid(iter); iter = m_typeMultiMap.getNext(iter, hash))
	{
		Type* curType = m_typeMultiMap.getValue(iter);

		if (curType == type)
		{
			return true;
		}
	}

	return false;
}

hkTypeManager::Type* hkTypeManager::copyType(Type* type)
{
	if (isOwned(type))
	{
		return type;
	}

	// Rebuild the type

	hkInplaceArray<Type*, 16> path;
	Type::getTypePath(type, path);

	Type* srcRoot = path.back();

	HK_ASSERT(0x32423a32, srcRoot->getParent() == HK_NULL);

	// The root can only be a terminal (int,real,class)
	Type* dstCur = HK_NULL;
	if (srcRoot->isClass())
	{
		dstCur = addClass(srcRoot->getTypeName());
	}
	else
	{
		dstCur = getSubType(srcRoot->getSubType());
	}

	HK_ASSERT(0x324a3a24, dstCur);
	if (!dstCur)
	{
		return HK_NULL;
	}

	for (int i = path.getSize() - 2; i >= 0; i--)
	{
		dstCur = replaceParent(path[i], dstCur);
	}

	return dstCur;
}


/* static */void HK_CALL hkTypeManager::appendTypeExpression(const Type* type, hkOstream& stream)
{
	while (type)
	{
		switch (type->getSubType())
		{
			case SUB_TYPE_INVALID:		stream << "!"; break;
			case SUB_TYPE_VOID:			stream << "v"; break;
			case SUB_TYPE_BYTE:			stream << "b"; break;
			case SUB_TYPE_REAL:			stream << "r"; break;
			case SUB_TYPE_INT:			stream << "i"; break;
			case SUB_TYPE_CSTRING:		stream << "s"; break;
			case SUB_TYPE_CLASS:		stream << "C" << type->getTypeName() << ";"; break;
			case SUB_TYPE_POINTER:		stream << "*"; break;
			case SUB_TYPE_ARRAY:		stream << "[]"; break;
			case SUB_TYPE_TUPLE:		stream << "{" << type->getTupleSize() << "}"; break;
			case SUB_TYPE_COUNT_OF: break;
		}

		type = type->getParent();
	}
}

hkTypeManager::Type* hkTypeManager::parseTypeExpression(const char* typeExpression)
{
	switch (*typeExpression)
	{
		case '!':	return getSubType(SUB_TYPE_INVALID);
		case 'v':	return getSubType(SUB_TYPE_VOID);
		case 'b':	return getSubType(SUB_TYPE_BYTE);
		case 'r':	return getSubType(SUB_TYPE_REAL);
		case 'i':	return getSubType(SUB_TYPE_INT);
		case 's':	return getSubType(SUB_TYPE_CSTRING);
		case '*':
		{
			Type* parent = parseTypeExpression(typeExpression + 1);
			return makePointer(parent);
		}
		case '[':
		{
			HK_ASSERT(0x324243a2, typeExpression[1] == ']');
			Type* parent = parseTypeExpression(typeExpression + 2);
			return makeArray(parent);
		}
		case '{':
		{
			const char* start = typeExpression + 1;
			const char* cur = start;
			while (*cur >= '0' && *cur <= '9') cur++;

			char buffer[10];
			if ((int(cur - start) <= 0) || (int(cur - start) > int(HK_COUNT_OF(buffer)) - 1) || 
				(*cur != '}'))
			{
				HK_ASSERT(0x242343, !"Couldn't parse tuple size");
				return HK_NULL;
			}

			const int len = int(cur - start);
			hkString::strNcpy(buffer, start, len);
			buffer[len] = 0;
			const int size = hkString::atoi(buffer);

			Type* parent = parseTypeExpression(cur + 1);
			return makeTuple(parent, size);
		}
		case 'C':
		{
			const char* start = typeExpression + 1;
			const char* cur = start;
			while (true)
			{
				const char c = *cur;
				if ((c >= 'a' && c <= 'z') ||
					(c >= 'A' && c <= 'Z') ||
					(c >= '0' && c <= '9') ||
					(c == '_') ||
					(c == ':'))
				{
					cur++;
					continue;
				}
				break;
			}
			
			// Need to extract the name
			char buffer[128];
			if (int(cur - start) <= 0 || int(cur - start) > int(HK_COUNT_OF(buffer)) - 1 || *cur != ';')
			{
				HK_ASSERT(0x3243a2a4, !"Unable to parse class name");
				return HK_NULL;
			}

			const int len = int(cur - start);
			hkString::strNcpy(buffer, start, len);
			buffer[len] = 0;
			return addClass(buffer);
		}
		default:
		{
			HK_ASSERT(0x243243, !"Unknown expression character");
			return HK_NULL;
		}
	}
}

hkTypeManager::Type* hkTypeManager::getType(hkLegacyType::Type type, const char* className, int tupleSize)
{
	typedef hkLegacyType::Type LegacyType;
	LegacyType basicType = LegacyType(hkLegacyType::TYPE_MASK_BASIC_TYPES & type);

	Type* rootType = HK_NULL;
	switch (basicType)
	{
		case hkLegacyType::TYPE_VOID:		rootType = getSubType(SUB_TYPE_VOID); break;
		case hkLegacyType::TYPE_BYTE:		rootType = getSubType(SUB_TYPE_BYTE); break;
		case hkLegacyType::TYPE_INT:		rootType = getSubType(SUB_TYPE_INT); break;
		case hkLegacyType::TYPE_REAL:		rootType = getSubType(SUB_TYPE_REAL); break;
		case hkLegacyType::TYPE_CSTRING:	rootType = getSubType(SUB_TYPE_CSTRING); break;
		case hkLegacyType::TYPE_VEC_4:
		{
			rootType = makeTuple(getSubType(SUB_TYPE_REAL), 4);
			break;
		}
		case hkLegacyType::TYPE_VEC_8:
		{
			rootType = makeTuple(getSubType(SUB_TYPE_REAL), 8);
			break;
		}
		case hkLegacyType::TYPE_VEC_12:
		{
			rootType = makeTuple(getSubType(SUB_TYPE_REAL), 12);
			break;
		}
		case hkLegacyType::TYPE_VEC_16:
		{
			rootType = makeTuple(getSubType(SUB_TYPE_REAL), 16);
			break;
		}
		case hkLegacyType::TYPE_OBJECT:
		{
			if (className)
			{
				rootType = makePointer(addClass(className));
			}
			else
			{
				rootType = makePointer(getHomogenousClass());
			}
			break;
		}
		case hkLegacyType::TYPE_STRUCT:
		{
			if (className)
			{
				rootType = addClass(className); 
			}
			else
			{
				rootType = getHomogenousClass();
			}
			break;
		}
		default: break;
	}

	if (!rootType)
	{
		HK_ASSERT(0x2434332, !"Couldn't setup type");
		return HK_NULL;
	}
		
	if (type & hkLegacyType::TYPE_ARRAY)
	{
		return makeArray(rootType);
	}

	if (type & hkLegacyType::TYPE_TUPLE)
	{
		return makeTuple(rootType, tupleSize);
	}

	return rootType;
}

/* static */hkLegacyType::Type HK_CALL hkTypeManager::getTerminalLegacyType(Type* type, const char** classNameOut)
{
	*classNameOut = HK_NULL;
	switch (type->getSubType())
	{
		case SUB_TYPE_VOID:					return hkLegacyType::TYPE_VOID;
		case SUB_TYPE_BYTE:					return hkLegacyType::TYPE_BYTE;
		case SUB_TYPE_INT:					return hkLegacyType::TYPE_INT;
		case SUB_TYPE_REAL:					return hkLegacyType::TYPE_REAL;
		case SUB_TYPE_CSTRING:				return hkLegacyType::TYPE_CSTRING;
		case SUB_TYPE_CLASS:				
		{
			*classNameOut = type->getTypeName();
			return hkLegacyType::TYPE_STRUCT;
		}
		case SUB_TYPE_POINTER:			
		{
			Type* parent = type->getParent();
			if (parent->isClass())
			{
				*classNameOut = parent->getTypeName();
				return hkLegacyType::TYPE_OBJECT;
			}
			break;
		}
		case SUB_TYPE_TUPLE:
		{
			if (type->getParent()->isReal())
			{
				switch (type->getTupleSize())
				{
					case 4:		return hkLegacyType::TYPE_VEC_4;
					case 8:		return hkLegacyType::TYPE_VEC_8;
					case 12:	return hkLegacyType::TYPE_VEC_12;
					case 16:	return hkLegacyType::TYPE_VEC_16;
					break;
				}
			}
			break;
		}
		default:		
		{
			return hkLegacyType::TYPE_VOID;
		}
	}

	return hkLegacyType::TYPE_VOID;
}
			
/* static */hkLegacyType::Type HK_CALL hkTypeManager::getLegacyType(Type* type, const char** className, int& numTupleOut)
{
	*className = HK_NULL;
	numTupleOut = 0;

	switch (type->getSubType())
	{
		case SUB_TYPE_ARRAY:
		{
			return hkLegacyType::Type(getTerminalLegacyType(type->getParent(), className) | hkLegacyType::TYPE_ARRAY);
		}
		case SUB_TYPE_TUPLE:
		{
			if (type->isVec())
			{
				return getTerminalLegacyType(type, className);
			}
			numTupleOut = type->getTupleSize();
			return hkLegacyType::Type(getTerminalLegacyType(type->getParent(), className) | hkLegacyType::TYPE_TUPLE);
		}
		default:
		{
			return getTerminalLegacyType(type, className);
		}
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
