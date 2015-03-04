/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>

// this
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerPointerMapLayoutHandler.h>
#include <Common/Base/Container/PointerMap/hkPointerMap.h>
#include <Common/Base/Container/PointerMultiMap/hkPointerMultiMap.h>

#define STORAGE_SIZE(TYPE_SIZE) ((TYPE_SIZE == 8) ? sizeof(hkPointerMapStorage<8>::Type) : sizeof(hkPointerMapStorage<4>::Type))

// checks used to make sure that the behavior is consistent with hkPointerMapStorage.
HK_COMPILE_TIME_ASSERT( sizeof(hkPointerMapStorage<1>::Type) == STORAGE_SIZE(1) );
HK_COMPILE_TIME_ASSERT( sizeof(hkPointerMapStorage<2>::Type) == STORAGE_SIZE(2) );
HK_COMPILE_TIME_ASSERT( sizeof(hkPointerMapStorage<3>::Type) == STORAGE_SIZE(3) );
HK_COMPILE_TIME_ASSERT( sizeof(hkPointerMapStorage<4>::Type) == STORAGE_SIZE(4) );
HK_COMPILE_TIME_ASSERT( sizeof(hkPointerMapStorage<5>::Type) == STORAGE_SIZE(5) );
HK_COMPILE_TIME_ASSERT( sizeof(hkPointerMapStorage<6>::Type) == STORAGE_SIZE(6) );
HK_COMPILE_TIME_ASSERT( sizeof(hkPointerMapStorage<7>::Type) == STORAGE_SIZE(7) );
HK_COMPILE_TIME_ASSERT( sizeof(hkPointerMapStorage<8>::Type) == STORAGE_SIZE(8) );

// Pointer maps store the pointers or integers according to the following rule:
// - Internal storage is hkUlong (same size as void*) if the type is not 8 bytes long
// - Internal storage is hkUint64 if the type is 8 bytes long
static HK_FORCE_INLINE hk_size_t getStorageSize(hk_size_t typeSize)
{
	return STORAGE_SIZE(typeSize);
}

void hkTrackerPointerMapLayoutHandler::getReferences(
	hkTrackerLayoutBlock* curBlock,
	const void* curData,
	const hkTrackerTypeTreeNode* curType,
	hkTrackerLayoutCalculator* layoutCalc,
	hkArray<const hkTrackerLayoutBlock*>::Temp& newBlocks )
{
	// hkPointerMap uses an hkMap internally, which inherits from hkMapBase containing the actual data.
	// hkPointerMultiMap uses an hkMultiMap internally which does NOT inherit from hkMapBase.
	// we only have two cases for pointer map and pointer multimap, either both the key and the value are
	// 4 bytes long or they are both 8 bytes long
	hkTrackerTypeTreeCache* typeCache = layoutCalc->getTypeCache();
	const hkTrackerTypeTreeNode* keyType = curType->m_contains;
	const hkTrackerTypeTreeNode* valueType = keyType->m_next;
	bool keyIsPointer = (keyType->m_type == hkTrackerTypeTreeNode::TYPE_POINTER);
	bool valueIsPointer = (valueType->m_type == hkTrackerTypeTreeNode::TYPE_POINTER);

	if(curType->m_name == "hkPointerMultiMap")
	{
		const hkPointerMultiMap<void*, void*>* dummyMap = static_cast< const hkPointerMultiMap<void*, void*>* >(curData);
		const void* bufferPtr = dummyMap->getMemStart();
		hk_size_t bufferSize = dummyMap->getMemSize();

		if(bufferSize > 0)
		{
			// add reference to buffer to the current block
			curBlock->m_references.pushBack(bufferPtr);
			// add a new block corresponding to the body of the map
			const hkTrackerTypeTreeNode* bufferType = 
				typeCache->newNamedNode(hkTrackerTypeTreeNode::TYPE_NAMED, "buffer_hkPointerMultiMap", false);
			hkTrackerLayoutBlock* bufferBlock = new hkTrackerLayoutBlock(bufferType, bufferPtr, bufferSize);
			newBlocks.pushBack(bufferBlock);

			if(keyIsPointer || valueIsPointer)
			{
				const hk_size_t keySize = getStorageSize(layoutCalc->calcTypeSize(keyType));
				HK_ON_DEBUG(const hk_size_t valueSize = getStorageSize(layoutCalc->calcTypeSize(valueType));)
				HK_ASSERT(0x31eeda3f, keySize == valueSize);
				if(keySize == 4) // valueSize will be 4 as well
				{
					const hkPointerMultiMap<hkUint32, hkUint32>* map = static_cast< const hkPointerMultiMap<hkUint32, hkUint32>* >(curData);
					for( hkPointerMultiMap<hkUint32, hkUint32>::Iterator it = map->getIterator();
						map->isValid(it);
						it = map->getNext(it) )
					{
						if(keyIsPointer)
						{
							HK_ASSERT(0x1cb3c269, sizeof(void*) == sizeof(hkUint32));
							hkUint32 key = map->getKey(it);
							bufferBlock->m_references.pushBack(reinterpret_cast<void*>(static_cast<hkUlong>(key)));
						}
						if(valueIsPointer)
						{
							HK_ASSERT(0x122f417, sizeof(void*) == sizeof(hkUint32));
							hkUint32 value = map->getValue(it);
							bufferBlock->m_references.pushBack(reinterpret_cast<void*>(static_cast<hkUlong>(value)));
						}
					}
				}
				else // keysize == 8 and valueSize == 8
				{
					const hkPointerMultiMap<hkUint64, hkUint64>* map = static_cast< const hkPointerMultiMap<hkUint64, hkUint64>* >(curData);
					for( hkPointerMultiMap<hkUint64, hkUint64>::Iterator it = map->getIterator();
						map->isValid(it);
						it = map->getNext(it) )
					{
						if(keyIsPointer)
						{
							HK_ASSERT(0x145ad0ab, sizeof(void*) == sizeof(hkUint64));
							hkUint64 key = map->getKey(it);
							bufferBlock->m_references.pushBack(reinterpret_cast<void*>(static_cast<hkUlong>(key)));
						}
						if(valueIsPointer)
						{
							HK_ASSERT(0x45f6f603, sizeof(void*) == sizeof(hkUint64));
							hkUint64 value = map->getValue(it);
							bufferBlock->m_references.pushBack(reinterpret_cast<void*>(static_cast<hkUlong>(value)));
						}
					}
				}
			}
		}
	}
	else // hkPointerMap
	{
		const hkPointerMap<void*, void*>* dummyMap = static_cast< const hkPointerMap<void*, void*>* >(curData);
		const void* bufferPtr = dummyMap->getMemStart();
		hk_size_t bufferSize = dummyMap->getMemSize();

		if(bufferSize > 0)
		{
			// add reference to buffer to the current block
			curBlock->m_references.pushBack(bufferPtr);
			// add a new block corresponding to the body of the map
			const hkTrackerTypeTreeNode* bufferType = 
				typeCache->newNamedNode(hkTrackerTypeTreeNode::TYPE_NAMED, "buffer_hkPointerMap", false);
			hkTrackerLayoutBlock* bufferBlock = new hkTrackerLayoutBlock(bufferType, bufferPtr, bufferSize);
			newBlocks.pushBack(bufferBlock);

			if(keyIsPointer || valueIsPointer)
			{
				const hk_size_t keySize = getStorageSize(layoutCalc->calcTypeSize(keyType));
				HK_ON_DEBUG(const hk_size_t valueSize = getStorageSize(layoutCalc->calcTypeSize(valueType));)
				HK_ASSERT(0xd9392b7, keySize == valueSize);
				if(keySize == 4) // valueSize will be 4 as well
				{
					const hkPointerMap<hkUint32, hkUint32>* map = static_cast< const hkPointerMap<hkUint32, hkUint32>* >(curData);
					for( hkPointerMap<hkUint32, hkUint32>::Iterator it = map->getIterator();
						map->isValid(it);
						it = map->getNext(it) )
					{
						if(keyIsPointer)
						{
							HK_ASSERT(0xca86bed, sizeof(void*) == sizeof(hkUint32));
							hkUint32 key = map->getKey(it);
							bufferBlock->m_references.pushBack(reinterpret_cast<void*>(static_cast<hkUlong>(key)));
						}
						if(valueIsPointer)
						{
							HK_ASSERT(0x41630165, sizeof(void*) == sizeof(hkUint32));
							hkUint32 value = map->getValue(it);
							bufferBlock->m_references.pushBack(reinterpret_cast<void*>(static_cast<hkUlong>(value)));
						}
					}
				}
				else // keysize == 8 and valueSize == 8
				{
					const hkPointerMap<hkUint64, hkUint64>* map = static_cast< const hkPointerMap<hkUint64, hkUint64>* >(curData);
					for( hkPointerMap<hkUint64, hkUint64>::Iterator it = map->getIterator();
						map->isValid(it);
						it = map->getNext(it) )
					{
						if(keyIsPointer)
						{
							HK_ASSERT(0x214de938, sizeof(void*) == sizeof(hkUint64));
							hkUint64 key = map->getKey(it);
							bufferBlock->m_references.pushBack(reinterpret_cast<void*>(static_cast<hkUlong>(key)));
						}
						if(valueIsPointer)
						{
							HK_ASSERT(0xfe6f500, sizeof(void*) == sizeof(hkUint64));
							hkUint64 value = map->getValue(it);
							bufferBlock->m_references.pushBack(reinterpret_cast<void*>(static_cast<hkUlong>(value)));
						}
					}
				}
			}
		}
	}
}

hk_size_t hkTrackerPointerMapLayoutHandler::getSize(
	const hkTrackerTypeTreeNode* curType,
	hkTrackerLayoutCalculator* layoutCalc )
{
	if(curType->m_name == "hkPointerMultiMap")
	{
		return sizeof(hkPointerMultiMap<void*, void*>);
	}
	else // hkPointerMap
	{
		return sizeof(hkPointerMap<void*, void*>);
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
