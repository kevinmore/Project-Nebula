/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Packfile/hkPackfileData.h>

#include <Common/Base/Reflection/Registry/hkClassNameRegistry.h>
#include <Common/Base/Reflection/Registry/hkTypeInfoRegistry.h>
#include <Common/Base/Reflection/Attributes/hkAttributes.h>

#if 0 && defined(HK_DEBUG)
#	include <Common/Base/Fwd/hkcstdio.h>
	using namespace std;
#	define TRACE(A) A
#else
#	define TRACE(A) // nothing
#endif

hkPackfileData::hkPackfileData(const hkClassNameRegistry* reg) :
	m_topLevelObject(HK_NULL),
	m_name(HK_NULL),
	m_destructorsEnabled(true)
{
	if (reg)
	{
		m_packfileClassRegistry = reg;
	}
	else
	{
		m_packfileClassRegistry = hkBuiltinTypeRegistry::getInstance().getClassNameRegistry();
	}
}

void hkPackfileData::callDestructors()
{
	if( m_destructorsEnabled && m_trackedTypes.getSize() > 0 )
	{
		for( TrackedObjectMap::Iterator it = m_trackedObjects.getIterator();
			m_trackedObjects.isValid(it); it = m_trackedObjects.getNext(it) )
		{
			if( const hkTypeInfo* typeInfo = m_trackedTypes.getWithDefault(m_trackedObjects.getValue(it), HK_NULL) )
			{
				typeInfo->cleanupLoadedObject(m_trackedObjects.getKey(it));
			}
			else
			{
				HK_ASSERT3(0x4a4de329, false, "Can not call destructor for unregistered type '" << m_trackedObjects.getValue(it) << "'.");
			}
		}
	}
	m_topLevelObject = HK_NULL;
	m_trackedObjects.clear();
	m_trackedTypes.clear();
}

hkPackfileData::~hkPackfileData()
{
	callDestructors();

	int i;
	for( i = 0; i < m_memory.getSize(); ++i )
	{
		hkDeallocate(m_memory[i]);
	}
	for( i = 0; i < m_chunks.getSize(); ++i )
	{
		hkDeallocateChunk( (char*)m_chunks[i].pointer, m_chunks[i].numBytes, m_chunks[i].memClass );
	}
	hkDeallocate<char>(m_name);
}

void hkPackfileData::setName(const char* n)
{
	hkDeallocate<char>(m_name);
	m_name = hkString::strDup(n);
}

void hkPackfileData::getImportsExports( hkArray<Import>& impOut, hkArray<Export>& expOut ) const
{
	impOut = m_imports;
	expOut = m_exports;
}

void hkPackfileData::addExport( const char* symbolName, void* object )
{
	Export& e = m_exports.expandOne();
	e.name = symbolName;
	e.data = object;
}

void hkPackfileData::removeExport( void* object )
{
	for( int i = m_exports.getSize() - 1; i >= 0; --i )
	{
		if( m_exports[i].data == object )
		{
			m_exports.removeAt(i);
			// no early out: object may be exported under several names
		}
	}
}

void hkPackfileData::addImport( const char* symbolName, void** location )
{
#	if defined(HK_DEBUG)
	for( int i = 0; i < m_imports.getSize(); ++i )
	{
		HK_ASSERT2( 0x80fd563f, m_imports[i].location != location, "Duplicate import location found");
	}
#	endif
	Import& imp = m_imports.expandOne();
	imp.name = symbolName;
	imp.location = location;
}


void hkPackfileData::removeImport( void** location )
{
	for( int i = m_imports.getSize() - 1; i >= 0; --i )
	{
		if( m_imports[i].location == location )
		{
			m_imports.removeAt(i);
			// early out: location is unique
			break;
		}
	}
}

hkBool32 hkPackfileData::finishedObjects() const
{
	return m_trackedTypes.getSize() != 0;
}

static inline bool baseOrSameClass(const char* typeName, const char* topLevelClassName, const hkClassNameRegistry* classReg)
{
	HK_ASSERT(0x53ad1bc5, typeName && topLevelClassName);
	HK_ASSERT(0x58b56f4e, classReg);
	const hkClass* topLevelClass = classReg->getClassByName(topLevelClassName);
	HK_ASSERT3(0x720938bc, topLevelClass, "The class " << topLevelClassName << " is not registered.");
	const hkClass* baseClass = classReg->getClassByName(typeName);
	HK_ASSERT(0x11efc014, baseClass);
	return baseClass->isSuperClass(*topLevelClass);
}

void* hkPackfileData::getContentsPointer(const char* typeName, const hkTypeInfoRegistry* typeRegistry) const
{
	if( m_topLevelObject )
	{
		const char* topLevelClassName = m_trackedObjects.getWithDefault(m_topLevelObject, HK_NULL);
		HK_ASSERT(0x4a45f945, topLevelClassName);
		if( !typeName
			|| baseOrSameClass(typeName, topLevelClassName, m_packfileClassRegistry) )
		{
			if( !finishedObjects() && typeRegistry )
			{
				hkBool hasUnregisteredType = false;
				for( TrackedObjectMap::Iterator iter = m_trackedObjects.getIterator(); m_trackedObjects.isValid(iter); iter = m_trackedObjects.getNext(iter) )
				{
					void* obj = m_trackedObjects.getKey(iter);
					const char* className = m_trackedObjects.getValue(iter);
					if( const hkTypeInfo* typeInfo = typeRegistry->finishLoadedObject(obj, className) )
					{
						m_trackedTypes.insert(className, typeInfo);
					}
					else
					{
						HK_ASSERT3(0x4a45f946, false, "Found unregistered type '" << className << "'.");
						hasUnregisteredType = true;
					}
				}
				if( hasUnregisteredType )
				{
					return HK_NULL;
				}

				// Call any post finish functions
				for( hkArray<const hkVariant>::iterator it = m_postFinishObjects.begin(); it < m_postFinishObjects.end(); it++ )
				{
					void* ptr = it->m_object;
					const hkClass& klass = *it->m_class;
					const hkVariant* attr = klass.getAttribute("hk.PostFinish");
					HK_ASSERT2(0x1e974825, attr && (attr->m_class == &hkPostFinishAttributeClass), "Object does not have PostFinish attribute");
					const hkPostFinishAttribute* postFinishAttr = reinterpret_cast<hkPostFinishAttribute*>(attr->m_object);
					postFinishAttr->m_postFinishFunction(ptr);
				}
			}
			return m_topLevelObject;
		}
		else
		{
			HK_WARN(0x599a0b9c, "Requested '" << typeName << "', but packfile data contains '" << topLevelClassName << "'.");
		}
	}
	else
	{
		HK_WARN(0x599a0a9e, "The packfile data object does not have any content. May be packfile content was not versioned before access it?");
	}
	return HK_NULL;
}

void hkPackfileData::setContentsWithName(void* topLevelObject, const char* typeName)
{
	HK_ASSERT(0x045d23e1, topLevelObject != HK_NULL && typeName != HK_NULL);
	HK_ON_DEBUG(const char* topLevelClassName = m_trackedObjects.getWithDefault(topLevelObject, HK_NULL));
	HK_ASSERT2(0x045d23e2, topLevelClassName != HK_NULL, "Can not set the top level object. The provided pointer is not owned by the packfile data object.");
	HK_ASSERT3(0x045d23e3, baseOrSameClass(typeName, topLevelClassName, m_packfileClassRegistry), "Can not set the top level object. The provided type '" << typeName << "' is not a base type for '" << topLevelClassName << "'");
	m_topLevelObject = topLevelObject;
}

const char* hkPackfileData::getContentsTypeName() const
{
	if( m_topLevelObject )
	{
		const char* topLevelClassName = m_trackedObjects.getWithDefault(m_topLevelObject, HK_NULL);
		HK_ASSERT(0x4a45f945, topLevelClassName);
		return topLevelClassName;
	}
	return HK_NULL;
}

void hkPackfileData::setPackfileClassNameRegistry(const hkClassNameRegistry* classReg)
{
	m_packfileClassRegistry = classReg;
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
