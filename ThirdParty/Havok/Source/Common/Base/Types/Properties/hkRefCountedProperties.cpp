/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/Properties/hkRefCountedProperties.h>
#if defined(HK_PLATFORM_SPU)
#	include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#	include <Common/Base/Spu/Dma/Buffer/hkDmaBuffer.h>
#endif

#if !defined(HK_PLATFORM_SPU)

//
//	Constructor

hkRefCountedProperties::hkRefCountedProperties()
{}

//
//	Copy constructor

hkRefCountedProperties::hkRefCountedProperties(const hkRefCountedProperties& other)
{
	m_entries.append(other.m_entries);
}

//
//	Serialization constructor

hkRefCountedProperties::hkRefCountedProperties(hkFinishLoadedObjectFlag flag)
:	m_entries(flag)
{}

//
//	Destructor

hkRefCountedProperties::~hkRefCountedProperties()
{}

//
//	Adds a property to the collection. If a property is already installed for the given key, it will be replaced

void hkRefCountedProperties::addPropertyInternal(PropertyKey propertyKey, hkReferencedObject* propertyObject, ReferenceCountHandling referenceHandling)
{
	for (int i = m_entries.getSize() - 1; i >= 0; i--)
	{
		Entry& e = m_entries[i];
		if ( e.getKey() == propertyKey )
		{
			if ( referenceHandling == REFERENCE_COUNT_INCREMENT )
			{
				e.m_object = propertyObject;
			}
			else
			{
				e.m_object = HK_NULL;
				e.m_object.setAndDontIncrementRefCount( propertyObject );
			}
			return;
		}
	}

	// Did not find the object, must add new
	{
		Entry& e = m_entries.expandOne();
		e.setKey(propertyKey);
		if ( referenceHandling == REFERENCE_COUNT_INCREMENT )
		{
			e.m_object = propertyObject;
		}
		else
		{
			e.m_object.setAndDontIncrementRefCount( propertyObject );
		}
	}
}

//
//	Removes a property from the collection

void hkRefCountedProperties::removeProperty(PropertyKey propertyKey)
{
	for (int i = m_entries.getSize() - 1; i >= 0; i--)
	{
		if ( m_entries[i].getKey() == propertyKey )
		{
			m_entries.removeAt(i);
			return;
		}
	}
}

//
//	Replaces the property at the given key with the given object

void hkRefCountedProperties::replaceProperty(PropertyKey propertyKey, hkReferencedObject* newPropertyObject)
{
	for (int i = m_entries.getSize() - 1; i >= 0; i--)
	{
		if ( m_entries[i].getKey() == propertyKey )
		{
			m_entries[i].m_object = newPropertyObject;
		}
	}

	// Key not found, should not get here!
	HK_WARN_ALWAYS(0x1d11daed, "Failed to locate key " << propertyKey << " among the existing properties!");
}

#endif

//
//	Locates and returns the property at the given key. If no property was found, it will return null.

hkReferencedObject* hkRefCountedProperties::accessProperty(PropertyKey propertyKey) const
{
	const int numEntries				= m_entries.getSize();
	const Entry* HK_RESTRICT entries	= m_entries.begin();

	// Download the array from Ppu
#if defined(HK_PLATFORM_SPU)
	const int size = HK_NEXT_MULTIPLE_OF(16, sizeof(Entry) * numEntries);
	hkDynamicDmaBuffer entriesBuffer(size, HK_NULL);
	entriesBuffer.dmaGetAndWait(entries, hkSpuDmaManager::READ_COPY, HK_SPU_DMA_GROUP_STALL, size);
	entries	= reinterpret_cast<const Entry*>(entriesBuffer.getContents());
#endif

	hkReferencedObject* retObj = HK_NULL;
	for (int i = numEntries - 1; i >= 0; i--)
	{
		if ( entries[i].getKey() == propertyKey )
		{
			retObj = entries[i].m_object;
			break;
		}
	}

	// Return the object!
	return retObj;
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
