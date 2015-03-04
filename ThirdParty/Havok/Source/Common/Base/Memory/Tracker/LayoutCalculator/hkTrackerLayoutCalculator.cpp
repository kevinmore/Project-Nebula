/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/hkTrackerLayoutCalculator.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerArrayLayoutHandler.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerRefPtrLayoutHandler.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerStringMapLayoutHandler.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerStringPtrLayoutHandler.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerQueueLayoutHandler.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerPointerMapLayoutHandler.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerPadSpuLayoutHandler.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerJobQueueLayoutHandler.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerSetLayoutHandler.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerFlagsLayoutHandler.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerEnumLayoutHandler.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/hkTrackerExternalLayoutHandlerManager.h>
#include <Common/Base/Reflection/TypeTree/hkTrackerTypeTreeParser.h>
#include <Common/Base/Memory/Tracker/hkMemoryTracker.h>
#include <Common/Base/Config/hkConfigVersion.h>

hkTrackerLayoutCalculator::hkTrackerLayoutCalculator(hkTrackerTypeTreeCache* typeCache):
	m_typeCache(typeCache)
{
	// Add these

	// array handlers
	{
		hkTrackerLayoutHandler* handler = new hkTrackerArrayLayoutHandler;
		addHandler("hkArrayBase", handler);
		addHandler("hkArray", handler);
		addHandler("hkInplaceArray", handler);
		addHandler("hkSmallArray", handler);
		handler->removeReference();
	}
	// refptr handler
	{
		hkTrackerLayoutHandler* handler = new hkTrackerRefPtrLayoutHandler;
		addHandler("hkRefPtr", handler);
		handler->removeReference();
	}
	// stringptr handler
	{
		hkTrackerLayoutHandler* handler = new hkTrackerStringPtrLayoutHandler;
		addHandler("hkStringPtr", handler);
		handler->removeReference();
	}
	// padspu handler	 
	{
	 	hkTrackerLayoutHandler* handler = new hkTrackerPadSpuLayoutHandler;
	 	addHandler("hkPadSpu", handler);
	 	handler->removeReference();
	}
	// queue handler	 
	{
	 	hkTrackerLayoutHandler* handler = new hkTrackerQueueLayoutHandler;
	 	addHandler("hkQueue", handler);
	 	handler->removeReference();
	}
	// string map and storage string map handlers
	{
	 	hkTrackerLayoutHandler* handler = new hkTrackerStringMapLayoutHandler;
	 	addHandler("hkStringMap", handler);
	 	addHandler("hkStorageStringMap", handler);
	 	handler->removeReference();
	}
	// pointer map and pointer multi-map
	{
	 	hkTrackerLayoutHandler* handler = new hkTrackerPointerMapLayoutHandler;
	 	addHandler("hkPointerMap", handler);
	 	addHandler("hkPointerMultiMap", handler);
	 	handler->removeReference();
	}
	// hkJobQueue and hkJobQueueDynamicData handlers
	{
		hkTrackerLayoutHandler* handler = new hkTrackerJobQueueLayoutHandler;
		addHandler("hkJobQueue", handler);
		handler->removeReference();
		handler = new hkTrackerJobQueueDynamicDataLayoutHandler;
		addHandler("hkJobQueue::DynamicData", handler);
		handler->removeReference();
	}
	// hkSet
	{
		hkTrackerLayoutHandler* handler = new hkTrackerSetLayoutHandler;
		addHandler("hkSet", handler);
		handler->removeReference();
	}
	// hkFlags
	{
		hkTrackerLayoutHandler* handler = new hkTrackerFlagsLayoutHandler;
		addHandler("hkFlags", handler);
		handler->removeReference();
	}
	// hkEnum
	{
		hkTrackerLayoutHandler* handler = new hkTrackerEnumLayoutHandler;
		addHandler("hkEnum", handler);
		handler->removeReference();
	}

#if defined(HK_MEMORY_TRACKER_ENABLE)
	// Add externally registered handlers
	hkTrackerExternalLayoutHandlerManager& extHandlerManager = hkTrackerExternalLayoutHandlerManager::getInstance();
	extHandlerManager.addHandlersToLayoutCalculator(this);
#endif
	}

hkTrackerLayoutCalculator::~hkTrackerLayoutCalculator()
{
	clear();
}

hkBool hkTrackerLayoutCalculator::isDerived(const hkSubString& type, const hkSubString& baseClass) const
{
	if (type == baseClass)
	{
		return true;
	}

	hkMemoryTracker& tracker = hkMemoryTracker::getInstance();
	const hkMemoryTracker::ClassDefinition* clsDef = tracker.findClassDefinition(type);

	while (clsDef)
	{
		const char* parentTypeName = clsDef->m_parentTypeName;
		if (!parentTypeName)
		{
			break;
		}

		if (baseClass == parentTypeName)
		{
			return true;
		}

		clsDef = tracker.findClassDefinition(parentTypeName);
	}

	return false;
}

hkTrackerTypeLayout* hkTrackerLayoutCalculator::_createClassLayout(const Node* type)
{	
	if (type->m_type != Node::TYPE_NAMED && type->m_type != Node::TYPE_CLASS)
	{
		return HK_NULL;
	}

	// Look up the tracker
	hkMemoryTracker& tracker = hkMemoryTracker::getInstance();
	const hkMemoryTracker::TypeDefinition* typeDef = tracker.findTypeDefinition(type->m_name);
	if (!typeDef)
	{
		if (m_reportedType.hasKey(type))
		{
			return HK_NULL;
		}

		char buffer[256];

		hkOstream stream(buffer, HK_COUNT_OF(buffer), true);
		hkTrackerTypeTreeNode::dumpType(type, stream);

		HK_WARN(0x4234324, "Layout not found for type: " << buffer << "\n");
		m_reportedType.insert(type, 1);
		return HK_NULL;
	}

	if (typeDef->m_type == hkMemoryTracker::TypeDefinition::TYPE_BASIC)
	{
		// It's known - it has no members so doesn't need a layout
		return HK_NULL;
	}

	// Create the empty layout
	hkTrackerTypeLayout* layout = new hkTrackerTypeLayout(type, typeDef->m_alignment, typeDef->m_size);

	if (typeDef->m_type == hkMemoryTracker::ClassDefinition::TYPE_SCAN)
	{
		layout->m_fullScan = true;
		return layout;
	}
	HK_ASSERT(0x2344a2aa, typeDef->m_type == hkMemoryTracker::ClassDefinition::TYPE_CLASS);
	const hkMemoryTracker::ClassDefinition* clsDef = static_cast<const hkMemoryTracker::ClassDefinition*>(typeDef);

	layout->m_isVirtual = (clsDef->m_isVirtual != 0);

	int baseOffset = 0;
	while (clsDef)
	{
		// Go through the members
		for (int i = 0; i < clsDef->m_numMembers; i++)
		{
			const hkMemoryTracker::Member& srcMember = clsDef->m_members[i];

			// Work out the offset
			int offset = (baseOffset + srcMember.m_offset);

			const Node* memberType = hkTrackerTypeTreeParser::parseType(srcMember.m_typeName, *m_typeCache);
			if (!memberType)
			{
				continue;
			}

			hkTrackerTypeLayout::Member& dstMember = layout->m_members.expandOne();

			int flags = 0;
			if (srcMember.m_flags & hkMemoryTracker::Member::FLAG_BACK_POINTER)
			{
				flags |= hkTrackerTypeLayout::Member::FLAG_BACK_POINTER;
			}

			dstMember.m_type = memberType;
			dstMember.m_name = srcMember.m_name;
			dstMember.m_offset = hkUint16(offset);
			dstMember.m_size = hkUint16(srcMember.m_memberSize);
			dstMember.m_flags = flags;
		}

		if (clsDef->m_parentTypeName)
		{
			// Fix the parents pointer
			baseOffset += clsDef->m_parentOffset;
			// Get the parents type
			clsDef = tracker.findClassDefinition(clsDef->m_parentTypeName);
		}
		else
		{
			break;
		}
	}

	// Return the layout
	return layout;
}

void hkTrackerLayoutCalculator::clear()
{
	{
		hkStorageStringMap<hkTrackerLayoutHandler*>::Iterator iter = m_handlers.getIterator();
		for (; m_handlers.isValid(iter); iter = m_handlers.getNext(iter))
		{
			hkTrackerLayoutHandler* handler = m_handlers.getValue(iter);

			handler->removeReference();
		}
		m_handlers.clear();
	}

	{
		hkPointerMap<const Node*, const hkTrackerTypeLayout*>::Iterator iter = m_layoutMap.getIterator();
		for (; m_layoutMap.isValid(iter); iter = m_layoutMap.getNext(iter))
		{
			const hkTrackerTypeLayout* layout = m_layoutMap.getValue(iter);
			layout->removeReference();
		}
		m_layoutMap.clear();
	}
}

void hkTrackerLayoutCalculator::addHandler(const char* name, hkTrackerLayoutHandler* handler)
{
	hkStorageStringMap<hkTrackerLayoutHandler*>::Iterator iter = m_handlers.findKey(name);
	if (m_handlers.isValid(iter))
	{
		hkTrackerLayoutHandler* oldHandler = m_handlers.getValue(iter);
		oldHandler->removeReference();
	}

	// Add it
	handler->addReference();
	m_handlers.insert(name, handler);
}

hkTrackerLayoutHandler* hkTrackerLayoutCalculator::getHandler(const hkSubString& name) const
{
	hkInplaceArray<char, 128> buffer;
	const int len = name.length();
	buffer.setSize(len + 1);
	hkString::memCpy(buffer.begin(), name.m_start, len);
	buffer[len] = 0;
	return m_handlers.getWithDefault(buffer.begin(), HK_NULL);
}
	
void hkTrackerLayoutCalculator::setLayout(const Node* type, const hkTrackerTypeLayout* layout)
{
	HK_ASSERT(0x23423423, type->m_type == Node::TYPE_CLASS || type->m_type == Node::TYPE_CLASS_TEMPLATE ||
		type->m_type == Node::TYPE_NAMED);

	hkPointerMap<const Node*, const hkTrackerTypeLayout*>::Iterator iter = m_layoutMap.findKey(type);
	
	if (m_layoutMap.isValid(iter))
	{
		const hkTrackerTypeLayout* oldLayout = m_layoutMap.getValue(iter);
		oldLayout->removeReference();
	}

	layout->addReference();
	m_layoutMap.insert(type, layout);
}

const hkTrackerTypeLayout* hkTrackerLayoutCalculator::getLayout(const Node* type)
{
	const hkTrackerTypeLayout* layout = m_layoutMap.getWithDefault(type, HK_NULL);
	if (!layout)
	{
		layout = _createClassLayout(type);
		if (layout)
		{
			m_layoutMap.insert(type, layout);
		}
	}
	return layout;
}

/* static */hk_size_t hkTrackerLayoutCalculator::calcBasicSize(Node::Type type)
{
	switch (type)
	{
		case Node::TYPE_INT8:		return sizeof(hkInt8);
		case Node::TYPE_INT16:		return sizeof(hkInt16);
		case Node::TYPE_INT32:		return sizeof(hkInt32);
		case Node::TYPE_INT64:		return sizeof(hkInt64);
		case Node::TYPE_UINT8:		return sizeof(hkUint8);
		case Node::TYPE_UINT16:		return sizeof(hkUint16);
		case Node::TYPE_UINT32:		return sizeof(hkUint32);
		case Node::TYPE_UINT64:		return sizeof(hkUint64);
		case Node::TYPE_FLOAT32:	return sizeof(hkFloat32);
		case Node::TYPE_FLOAT64:	return sizeof(hkDouble64);
		case Node::TYPE_BOOL:		return sizeof(bool);		// Built in bool
		case Node::TYPE_POINTER:	return sizeof(void*);
		case Node::TYPE_REFERENCE:	return sizeof(void*&);
		default: return 0;
	}
}

hk_size_t hkTrackerLayoutCalculator::calcBasicAlignment(Node::Type type)
{
	switch (type)
	{
		case Node::TYPE_INT8:		return HK_ALIGN_OF(hkInt8);
		case Node::TYPE_INT16:		return HK_ALIGN_OF(hkInt16);
		case Node::TYPE_INT32:		return HK_ALIGN_OF(hkInt32);
		case Node::TYPE_INT64:		return HK_ALIGN_OF(hkInt64);
		case Node::TYPE_UINT8:		return HK_ALIGN_OF(hkUint8);
		case Node::TYPE_UINT16:		return HK_ALIGN_OF(hkUint16);
		case Node::TYPE_UINT32:		return HK_ALIGN_OF(hkUint32);
		case Node::TYPE_UINT64:		return HK_ALIGN_OF(hkUint64);
		case Node::TYPE_FLOAT32:	return HK_ALIGN_OF(hkFloat32);
		case Node::TYPE_FLOAT64:	return HK_ALIGN_OF(hkDouble64);
		case Node::TYPE_BOOL:		return HK_ALIGN_OF(bool);		// Built in bool
		case Node::TYPE_POINTER:	return HK_ALIGN_OF(void*);
		case Node::TYPE_REFERENCE:	return HK_ALIGN_OF(void*&);
		default: return 0;
	}
}

hkResult hkTrackerLayoutCalculator::calcTypeInfo(const Node* type, hkTrackerLayoutTypeInfo& typeInfo)
{
	switch (type->m_type)
	{
		default:
		{
			typeInfo.m_size = calcBasicSize(type->m_type);
			typeInfo.m_alignment = (int)calcBasicAlignment(type->m_type);
			if (typeInfo.m_size > 0 && typeInfo.m_alignment > 0)
			{
				return HK_SUCCESS;
			}
			return HK_FAILURE;
		}
		
		case Node::TYPE_POINTER:	
		{
			typeInfo.m_size = calcBasicSize(type->m_type);
			typeInfo.m_alignment = (int)calcBasicAlignment(type->m_type);
			return HK_SUCCESS;
		}
		case Node::TYPE_REFERENCE:	
		{
			typeInfo.m_size = calcBasicSize(type->m_type);
			typeInfo.m_alignment = (int)calcBasicAlignment(type->m_type);
			return HK_SUCCESS;
		}
		case Node::TYPE_ARRAY:
		{
			hkResult res = calcTypeInfo(type->m_contains, typeInfo);
			if (res == HK_FAILURE)
			{
				return res;
			}

			typeInfo.m_size *= type->m_dimension;
			return HK_SUCCESS;
		}
		case Node::TYPE_CLASS_TEMPLATE:
		{
			// templates are only supported using handlers
			return HK_FAILURE;
		}
		case Node::TYPE_NAMED:
		case Node::TYPE_CLASS:
		{
			const hkTrackerTypeLayout* layout = _getLayout(type);

			if (!layout)
			{
				hkMemoryTracker* tracker = &hkMemoryTracker::getInstance();
				const hkMemoryTracker::TypeDefinition* typeDef = tracker->findTypeDefinition(type->m_name);
				if (typeDef && typeDef->m_type == hkMemoryTracker::TypeDefinition::TYPE_BASIC)
				{
					typeInfo.m_size = typeDef->m_size;
					typeInfo.m_alignment = typeDef->m_alignment;
					return HK_SUCCESS;
				}

				// May as well create the layout then
				layout = getLayout(type);
			}

			if (layout)
			{
				typeInfo.m_size = layout->m_size;
				typeInfo.m_alignment = layout->m_alignment;
				return HK_SUCCESS;
			}

			return HK_FAILURE;
		}
		case Node::TYPE_ENUM:
		{
			hkMemoryTracker* tracker = &hkMemoryTracker::getInstance();
			const hkMemoryTracker::TypeDefinition* typeDef = tracker->findTypeDefinition(type->m_name);
			if (typeDef)
			{
				typeInfo.m_size = typeDef->m_size;
				typeInfo.m_alignment = typeDef->m_alignment;
				return HK_SUCCESS;				
			}
			return HK_FAILURE;
		}
	}
}

void hkTrackerLayoutCalculator::getReferencesRecursive( 
	hkTrackerLayoutBlock* block, 
	const void* curData,
	const hkTrackerTypeTreeNode* curType,
	hkArray<const hkTrackerLayoutBlock*>::Temp& newBlocks )
{
	switch(curType->m_type)
	{
	case hkTrackerTypeTreeNode::TYPE_CLASS:
	case hkTrackerTypeTreeNode::TYPE_CLASS_TEMPLATE:
	case hkTrackerTypeTreeNode::TYPE_NAMED:
		{
			// Use the handler if it has one
			hkTrackerLayoutHandler* handler = getHandler(curType->m_name);
			if (handler)
			{
				// If there is a handler
				hkArray<const hkTrackerLayoutBlock*>::Temp ownedBlocks;
				handler->getReferences(block, curData, curType, this, ownedBlocks);
				newBlocks.append(ownedBlocks);
				break; // done looking for references
			}

			// Run through the members if it is not a template
			if( curType->m_type == hkTrackerTypeTreeNode::TYPE_CLASS ||
				curType->m_type == hkTrackerTypeTreeNode::TYPE_NAMED )
			{
				const hkTrackerTypeLayout* layout = getLayout(curType);
				if(layout == HK_NULL)
				{
					// it either has no member or the layout could not be determined
					break;
				}
				// run through the members and find additional references
				for (int i = 0; i < layout->m_members.getSize(); i++)
				{
					const hkTrackerTypeLayout::Member& member = layout->m_members[i];

					const void* memberData = static_cast<const hkUint8*>(curData) + member.m_offset;

					// Recur
					getReferencesRecursive(block, memberData, member.m_type, newBlocks);
				}
				break; // done looking for references
			}

			#if (HAVOK_BUILD_NUMBER == 0) && defined(HK_DEBUG)
				hkStringBuf buf = "No handler was found for template class: ";
				buf.append(curType->m_name.m_start, curType->m_name.length());
				HK_ASSERT(0x463c1c10, buf.cString());
			#endif
		}
		break;
	case hkTrackerTypeTreeNode::TYPE_POINTER:
	case hkTrackerTypeTreeNode::TYPE_REFERENCE:
		{
			HK_ASSERT(0x4ae367df, curType->m_contains != HK_NULL);
			// void* are not considered to own references to anything.
			
			if(curType->m_contains->m_type != hkTrackerTypeTreeNode::TYPE_VOID)
			{
				void* ptr = *static_cast<void* const*>(curData);
				block->m_references.pushBack(ptr);
			}
		}
		break;
	case hkTrackerTypeTreeNode::TYPE_ARRAY:
		{
			const int dimension = curType->m_dimension;
			const hkTrackerTypeTreeNode* containedType = curType->m_contains;
			hk_size_t containedSize = calcTypeSize(containedType);
			const hkUint8* data = static_cast<const hkUint8*>(curData);

			for (int i = 0; i < dimension; i++, data += containedSize)
			{
				getReferencesRecursive(block, data, containedType, newBlocks);
			}
		}
		break;
	default:
		break;
	}
}

void hkTrackerLayoutCalculator::getReferences( 
	hkTrackerLayoutBlock* block, 
	hkArray<const hkTrackerLayoutBlock*>::Temp& newBlocks )
{
	// compute number of elements of the given type inside the block
	if (block->m_arraySize > 1)
	{
		hk_size_t size = calcTypeSize(block->m_type);
		HK_ASSERT(0x10a424ae, size != 0); // size should always be known
		const hkUint8* cur = (const hkUint8*)block->m_start;
		for (int i = 0; i < block->m_arraySize; i++, cur += size)
		{
			getReferencesRecursive(block, cur, block->m_type, newBlocks);
		}
	}
	else
	{
		getReferencesRecursive( block, block->m_start, block->m_type, newBlocks);
	}
}

hk_size_t hkTrackerLayoutCalculator::_calcTypeSizeFromMember(const Node* type)
{
	hkMemoryTracker* tracker = &hkMemoryTracker::getInstance();

	hk_size_t numTypes = tracker->getTypeDefinitions(HK_NULL);

	hkArray<const hkMemoryTracker::TypeDefinition*> types;
	types.setSize(int(numTypes));
	tracker->getTypeDefinitions(types.begin());

	for (int i = 0; i < types.getSize(); i++)
	{
		// Look at the members
		const hkMemoryTracker::TypeDefinition* typeDef = types[i];

		if (typeDef->m_type == hkMemoryTracker::TypeDefinition::TYPE_CLASS)
		{
			const hkMemoryTracker::ClassDefinition* clsDef = static_cast<const hkMemoryTracker::ClassDefinition*>(typeDef);

			for (int j = 0; j < clsDef->m_numMembers; j++)
			{
				const hkMemoryTracker::Member& member = clsDef->m_members[j];
				// If the member is the same type, we have its size
				if (type == hkTrackerTypeTreeParser::parseType(member.m_typeName, *m_typeCache))
				{
					return member.m_memberSize;
				}
			}
		}
	}

	return 0;
}

void hkTrackerLayoutCalculator::setTypeSize(const Node* type, hk_size_t size)
{
	m_sizeMap.insert(type, size);
}

hk_size_t hkTrackerLayoutCalculator::calcTypeSize(const Node* type)
{
	// Basic type
	hk_size_t size = calcBasicSize(type->m_type);
	if (size > 0)
	{
		return size;
	}

	// See if its been set
	size = m_sizeMap.getWithDefault(type, 0);
	if (size > 0)
	{
		return size;
	}

	// It's named, so look it up
	switch (type->m_type)
	{
		case Node::TYPE_ARRAY:
		{
			return calcTypeSize(type->m_contains) * type->m_dimension;
		}
		case Node::TYPE_CLASS_TEMPLATE:
		case Node::TYPE_NAMED:
		case Node::TYPE_CLASS:
		{
			// Use the handler if it has one
			hkTrackerLayoutHandler* handler = getHandler(type->m_name);
			if (handler)
			{
				size = handler->getSize(type, this);
				m_sizeMap.insert(type, size);
				return size;
			}
			
			hkTrackerLayoutTypeInfo typeInfo;
			hkResult res = hkTrackerLayoutCalculator::calcTypeInfo(type, typeInfo);
			if (res == HK_SUCCESS)
			{
				return typeInfo.m_size;
			}

			size = _calcTypeSizeFromMember(type);
			if (size > 0)
			{
				m_sizeMap.insert(type, size);
				return size;
			}

			// Urgh, okay perhaps I can find its size as a member of something else
			return 0;
		}
		default: break;
	}

	// Its not known
	return 0;
}

void hkTrackerLayoutCalculator::calcMembers(const Node* type, hk_size_t size, hkArray<hkTrackerTypeLayout::Member>& membersOut, int baseIndex, int flags)
{
	if (!type)
	{
		// If no type its just an unknown pointer
		hkTrackerTypeLayout::Member& dstMember = membersOut.expandOne();

		HK_ASSERT(0x32423423, size == sizeof(void*));

		dstMember.m_name = HK_NULL;
		dstMember.m_offset = hkUint16(baseIndex);
		dstMember.m_size = hkUint16(size);
		dstMember.m_type = type;
		dstMember.m_flags = flags;
		return;
	}

	switch (type->m_type)
	{
		case Node::TYPE_CLASS_TEMPLATE:
		case Node::TYPE_NAMED:
		case Node::TYPE_CLASS:
		{
			const hkTrackerTypeLayout* layout = getLayout(type);
			if (layout)
			{
				HK_ASSERT(0x2423423, layout->m_fullScan == false);
				
				for (int i = 0; i < layout->m_members.getSize(); i++)
				{
					const hkTrackerTypeLayout::Member& member = layout->m_members[i];
					calcMembers(member.m_type, member.m_size, membersOut, baseIndex + member.m_offset, flags | member.m_flags);
				}
			}
			break;
		}
		case Node::TYPE_POINTER:	
		case Node::TYPE_REFERENCE:
		{
			hkTrackerTypeLayout::Member& dstMember = membersOut.expandOne();

			dstMember.m_name = HK_NULL;
			dstMember.m_offset = hkUint16(baseIndex);
			dstMember.m_size = hkUint16(sizeof(void*));
			dstMember.m_type = type;
			dstMember.m_flags = flags;
			break;
		}
		case Node::TYPE_ARRAY:
		{
			const int dimension = type->m_dimension;
			const Node* containedType = type->m_contains;

			// I need to get the layout of the child
			hk_size_t containedSize = calcTypeSize(containedType);
			if (containedSize >= sizeof(void*))
			{
				int offset = baseIndex;
				for (int i = 0; i < dimension; i++, offset += (int)containedSize)
				{
					calcMembers(containedType, containedSize, membersOut, offset, flags);
				}
			}
			break;
		}
		default: 
		{
			break;
		}
	}
}

static void HK_CALL _concatMemberName(const char* name, int memberIndex, hkStringBuf& buf)
{
	if (buf.getLength() > 0)
	{
		buf.append(".");
	}

	if (name)
	{
		// The name is the bit before ::
		const char* start = name;
		const char* cur = start + hkString::strLen(start) - 1;
		while (cur >= start && *cur != ':') cur--;

		// Append just the name
		buf.append(cur + 1);
	}
	else
	{
		buf.appendPrintf("_unknown%i", memberIndex);
	}
}

static void HK_CALL _addName(hkStringBuf& baseName, hkArray<char>& buffer, hkArray<int>& namesOffset)
{
	// Concat the stuff before
	namesOffset.pushBack(buffer.getSize());

	// Write out the name
	char* dst = buffer.expandBy(baseName.getLength() + 1);
	hkString::memCpy(dst, baseName.cString(), baseName.getLength());
	buffer.back() = 0;
}

void hkTrackerLayoutCalculator::_calcMemberNames(const Node* type, hkStringBuf& baseName, hkArray<char>& buffer, hkArray<int>& namesOffset)
{
	if (!type)
	{
		_addName(baseName, buffer, namesOffset);
		return;
	}

	switch (type->m_type)
	{
		case Node::TYPE_CLASS_TEMPLATE:
		case Node::TYPE_NAMED:
		case Node::TYPE_CLASS:
		{
			const hkTrackerTypeLayout* layout = getLayout(type);

			if (layout)
			{
				// Don't support embedding of full scan types for now, as doing so would mean flattening wouldn't work
				HK_ASSERT(0x2423423, layout->m_fullScan == false);

				for (int i = 0; i < layout->m_members.getSize(); i++)
				{
					const hkTrackerTypeLayout::Member& member = layout->m_members[i];

					const int len = baseName.getLength();	
					_concatMemberName(member.m_name, i, baseName);
					_calcMemberNames(member.m_type, baseName, buffer, namesOffset);
					baseName.chompEnd(baseName.getLength() - len);
				}
			}
			break;
		}
		case Node::TYPE_POINTER:	
		case Node::TYPE_REFERENCE:
		{
			_addName(baseName, buffer, namesOffset);
			break;
		}
		case Node::TYPE_ARRAY:
		{
			const int dimension = type->m_dimension;
			const Node* containedType = type->m_contains;

			// I need to get the layout of the child
			hk_size_t containedSize = calcTypeSize(containedType);
			if (containedSize >= sizeof(void*))
			{
				for (int j = 0; j < dimension; j++)
				{
					const int len = baseName.getLength();

					baseName.append("[");
					baseName.appendPrintf("%i", j);
					baseName.append("]");

					_calcMemberNames(containedType, baseName, buffer, namesOffset);

					baseName.chompEnd(baseName.getLength() - len);
				}
			}
			break;
		}
		default: 
		{
			break;
		}
	}
}

void hkTrackerLayoutCalculator::calcMemberNames(const Node* type, hkArray<char>& buffer, hkArray<int>& namesOffset)
{
	hkStringBuf baseName;

	_calcMemberNames(type, baseName, buffer, namesOffset);
}

void hkTrackerLayoutCalculator::flattenTypes()
{
	hkPointerMap<const Node*, const hkTrackerTypeLayout*>::Iterator iter = m_layoutMap.getIterator();

	for (; m_layoutMap.isValid(iter); iter = m_layoutMap.getNext(iter))
	{
		const hkTrackerTypeLayout* srcLayout = m_layoutMap.getValue(iter);
		if (srcLayout->m_fullScan)
		{
			// If its full scan it has no members (and no member names)
			// If its flattened it has already been expanded.
			continue;
		}

		hkTrackerTypeLayout* dstLayout = new hkTrackerTypeLayout(srcLayout->m_type, srcLayout->m_alignment, srcLayout->m_size);

		dstLayout->m_isVirtual = srcLayout->m_isVirtual;

		// Work out the flattened names
		calcMembers(srcLayout->m_type, srcLayout->m_size, dstLayout->m_members, 0, 0);

		hkInplaceArray<int, 32> namesOffset;

		calcMemberNames(srcLayout->m_type, dstLayout->m_nameBuffer, namesOffset);

		HK_ASSERT(0x2332432, namesOffset.getSize() == dstLayout->m_members.getSize());

		for (int i = 0; i < namesOffset.getSize(); i++)
		{
			hkTrackerTypeLayout::Member& member = dstLayout->m_members[i];
			member.m_name = dstLayout->m_nameBuffer.begin() + namesOffset[i];
		}

		// Replace
		srcLayout->removeReference();
		m_layoutMap.setValue(iter, dstLayout);
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
