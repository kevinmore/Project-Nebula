/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/hkSerialize.h>
#include <Common/Compat/Deprecated/Version/hkPackfileObjectUpdateTracker.h>

hkPackfileObjectUpdateTracker::hkPackfileObjectUpdateTracker( hkPackfileData* data )
	: m_packfileData(data), m_topLevelObject(HK_NULL)
{
	m_packfileData->addReference();
}

hkPackfileObjectUpdateTracker::~hkPackfileObjectUpdateTracker()
{
	m_packfileData->removeReference();
}

void hkPackfileObjectUpdateTracker::setTopLevelObject( void* topLevel, const char* topClass )
{
	HK_ASSERT(0x53409994, topLevel!= HK_NULL);
	HK_ASSERT(0x53409995, topClass != HK_NULL);
	HK_ASSERT(0x53409996, m_topLevelObject == HK_NULL);
	m_topLevelObject = topLevel;
	m_finish.insert( topLevel, topClass); //like addFinish(topLevel, topClass), without assert
	objectPointedBy(m_topLevelObject, &m_topLevelObject);
}

const char* hkPackfileObjectUpdateTracker::getTopLevelClassName()
{
	HK_ASSERT(0x22151a55, m_topLevelObject != HK_NULL );
	return m_finish.getWithDefault( m_topLevelObject, HK_NULL );
}

void hkPackfileObjectUpdateTracker::addAllocation(void* p)
{
	m_packfileData->addAllocation(p);
}

void hkPackfileObjectUpdateTracker::addChunk(void* p, int n, HK_MEMORY_CLASS c)
{
	m_packfileData->addChunk(p,n,c);
}

void hkPackfileObjectUpdateTracker::objectPointedBy( void* newObject, void* fromWhere )
{
	HK_SERIALIZE_LOG(("TrackObjectPointedBy(obj=0x%p,loc=0x%p)\n", newObject, fromWhere));
	void* oldObject = *static_cast<void**>(fromWhere);
	if (oldObject)
	{
		for( int i = m_pointers.getFirstIndex(oldObject);
			i != -1;
			i = m_pointers.getNextIndex(i) )
		{
			if( m_pointers.getValue(i) == fromWhere )
			{
				if( newObject == oldObject )
				{
					return;
				}
				m_pointers.removeByIndex(oldObject, i);
				break;
			}
		}
	}
	if( newObject )
	{
		m_pointers.insert( newObject, fromWhere );
	}
	*static_cast<void**>(fromWhere) = newObject;
}

void hkPackfileObjectUpdateTracker::replaceObject( void* oldObject, void* newObject, const hkClass* newClass )
{
	HK_ASSERT(0x2ab05659, oldObject);
	HK_SERIALIZE_LOG(("TrackReplaceObject(oldObj=0x%p,newObj=0x%p,klassname=\"%s\")\n", oldObject, newObject, newClass ? newClass->getName() : ""));
	// replace pointers to old object with pointers to new one
	int oldObjectFirstIndex = m_pointers.getFirstIndex(oldObject);
	int newObjectFirstIndex = newObject ? m_pointers.getFirstIndex(newObject) : -1;

	for( int index = oldObjectFirstIndex; index != -1; index = m_pointers.getNextIndex(index) )
	{
		void* ptrOldObject = m_pointers.getValue(index);
		HK_ASSERT3(0x7fe24edd, *static_cast<void**>(ptrOldObject) == oldObject, "Expected 0x" << oldObject << " at 0x" << ptrOldObject << " (0x" << *static_cast<void**>(ptrOldObject) << ").");
		*static_cast<void**>(ptrOldObject) = newObject;

		// if newObject key already exists then we simply add values to it
		if( newObjectFirstIndex != -1 )
		{
#if defined(HK_DEBUG)
			for( int newObjectIndex = newObjectFirstIndex; newObjectIndex != -1; newObjectIndex = m_pointers.getNextIndex(newObjectIndex) )
			{
				// relink to existing newObject key and normalize
				void* ptrNewObject = m_pointers.getValue(newObjectIndex);
				HK_ASSERT(0x6f42cb23, ptrNewObject != ptrOldObject);
			}
#endif
			//HK_REPORT("Relink 0x" << ptrOldObject << " (0x" << oldObject << ") to 0x" << newObject << ".");
			m_pointers.insert(newObject, ptrOldObject);
		}
	}
	if( newObjectFirstIndex != -1 || newObject == HK_NULL ) // if newObject key already exists or it is HK_NULL then we may remove oldObject key now
	{
		m_pointers.removeKey(oldObject);
		//HK_REPORT("Remove key 0x" << oldObject << ".");
	}
	else if( oldObjectFirstIndex != -1 ) // if only oldObject key exists and newObject is new key
	{
		//HK_REPORT("Replace key 0x" << oldObject << " -> 0x" << newObject << ".");
		m_pointers.changeKey(oldObject, newObject);
	}
	// keep exports valid
	for( int i = 0; i < m_packfileData->m_exports.getSize(); ++i )
	{
		if( m_packfileData->m_exports[i].data == oldObject )
		{
			m_packfileData->m_exports[i].data = newObject;
		}
	}
	// replace object in the finish list
	removeFinish(oldObject);
	if( newClass )
	{
		addFinish(newObject, newClass->getName());
	}
}

void hkPackfileObjectUpdateTracker::addFinish( void* newObject, const char* className )
{
	HK_ASSERT( 0x567037f2, m_finish.hasKey(newObject) == false );
	m_finish.insert( newObject, className );
}

void hkPackfileObjectUpdateTracker::removeFinish( void* oldObject )
{
	m_finish.remove(oldObject);
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
