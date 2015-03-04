/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/ResourceDatabase/hkResourceHandle.h>
#include <Common/Base/Container/String/hkStringBuf.h>

hkMemoryResourceHandle::hkMemoryResourceHandle()
: hkResourceHandle()
{
}


hkMemoryResourceHandle::hkMemoryResourceHandle(hkFinishLoadedObjectFlag flag)
:	hkResourceHandle(flag), m_variant(flag), m_name(flag), m_references(flag)
{
}


hkMemoryResourceHandle::~hkMemoryResourceHandle()
{
}


void hkMemoryResourceHandle::clearExternalLinks()
{
	m_references.clear();
}


const char* hkMemoryResourceHandle::getName(hkStringBuf& buffer) const
{
	if ( m_name )
	{
		return m_name;
	}
	return "null";
}


void hkMemoryResourceHandle::setName(const char* name)
{
	m_name = name;
}


void* hkMemoryResourceHandle::getObject( ) const
{
	return m_variant.val();
}


const hkClass* hkMemoryResourceHandle::getClass() const
{
	return m_variant.getClass();
}


void hkMemoryResourceHandle::setObject(void* object, const hkClass* klass)
{
	m_variant.set(object, klass);
}


void hkMemoryResourceHandle::addExternalLink(const char* memberName, const char* externalId)
{
	ExternalLink& link = m_references.expandOne();
	link.m_memberName            = memberName;
	link.m_externalId            = externalId;
}

void hkMemoryResourceHandle::removeExternalLink( const char* memberName )
{
	for (int i =0; i < m_references.getSize(); i++ )
	{
		if ( 0 == hkString::strCmp(m_references[i].m_memberName, memberName) )
		{
			m_references.removeAt(i);
			return;
		}
	}
}


void hkMemoryResourceHandle::getExternalLinks(hkArray<Link>& linksOut)
{
	linksOut.clear();
	linksOut.reserve(m_references.getSize());

	char buffer[512];

	{
		for (int i = 0; i < m_references.getSize(); i++)
		{
			Link& link = *linksOut.expandByUnchecked(1);
			const char* memberName = m_references[i].m_memberName;

			void* object = m_variant.val();
			const hkClass* klass = m_variant.getClass();
			const hkClassMember* member;

			while(1)
			{
				const char* sep = hkString::strChr( memberName, '.');
				if ( !sep )
				{
					member = klass->getMemberByName( memberName );
					break;
				}
				int len = int(sep - memberName);
				hkString::memCpy( buffer, memberName, len );
				buffer[len] = 0;
				memberName += len + 1;

				member = klass->getMemberByName( buffer );
				if ( !member )
				{
					break;
				}
				hkClassMember::Type type = member->getType();
				if ( type != hkClassMember::TYPE_STRUCT )
				{
					HK_WARN_ALWAYS( 0xf032edfe, "Member is not of type struct : " << buffer );
					member = HK_NULL;
					break;
				}
				klass = member->getClass();
				object = hkAddByteOffset( object, member->getOffset() );
			}


			if ( member )
			{
				link.m_memberName     = memberName;
				link.m_memberAccessor = hkClassMemberAccessor(object, member);
				link.m_externalId     = m_references[i].m_externalId;
			}
			else
			{
				linksOut.popBack();
				HK_WARN_ALWAYS( 0xf032edf1, "Cannot find member : " << memberName );
			}
		}
	}
}


void hkResourceHandle::tryToResolveLinks(hkResourceMap& map)
{
	hkArray<Link> links;
	getExternalLinks( links );


	for (int i = links.getSize()-1; i >= 0; i--)
	{
		Link& link = links[i];

		const hkClass* externalClass;
		void* externalObject = map.findObjectByName(link.m_externalId, &externalClass);

		if ( externalObject == HK_NULL )
		{
			//HK_WARN(0xaf12e114, "Cannot resolve " << link.m_externalId);
			continue;
		}

		const hkClassMember& linkedClassMember = link.m_memberAccessor.getClassMember();
		const hkClass* linkedClass = linkedClassMember.getClass();

		// Ignore all objects whose type is not matching.
		if ( !linkedClass->isSuperClass(*externalClass) && ( externalClass != linkedClass ) )
		{
			HK_WARN_ALWAYS( 0xf034ed21, "Class mismatch, cannot resolve link: " << externalClass->getName() << " != " << linkedClass->getName() );
			continue;
		}

		void* object = getObject();
		void* linkedObject = externalObject;
		if ( linkedObject == object )
		{
			HK_WARN(0xaf12e113, "Circular dependency in linked object!");
			return;
		}

		link.m_memberAccessor.asPointer() = linkedObject;

		// Remove the reference as we have successfully resolved it.
		removeExternalLink( linkedClassMember.getName() );
	}
}

void hkResourceContainer::tryToResolveLinks(hkResourceMap& map)
{
	hkArray<hkResourceHandle*> handles; findAllResourceRecursively( handles );

	for ( int i = 0; i < handles.getSize(); i++  )
	{
		hkResourceHandle* handle = handles[i];
		handle->tryToResolveLinks(map);
	}
}

void hkResourceContainer::getPath(hkStringBuf& pathOut)
{
	hkResourceContainer* parent = getParent();
	if ( parent )
	{
		parent->getPath( pathOut );
	}
	hkStringBuf buffer;
	const char* name = getName( buffer );

	pathOut += "/";
	pathOut += name;
}


void hkResourceContainer::findAllResourceRecursively( hkArray<hkResourceHandle*>& resourcesOut )
{
	for (hkResourceContainer* container = findContainerByName(HK_NULL, HK_NULL); container;  container = findContainerByName(HK_NULL, container))
	{
		container->findAllResourceRecursively( resourcesOut );
	}

	for (hkResourceHandle* handle = findResourceByName(HK_NULL, HK_NULL, HK_NULL); handle;  handle = findResourceByName(HK_NULL, HK_NULL, handle))
	{
		resourcesOut.pushBack(handle);
	}
}

void hkResourceContainer::findAllContainersRecursively( hkArray<hkResourceContainer*>& resourcesOut )
{
	resourcesOut.pushBack( this );
	for (hkResourceContainer* container = findContainerByName(HK_NULL, HK_NULL); container;  container = findContainerByName(HK_NULL, container))
	{
		container->findAllContainersRecursively( resourcesOut );
	}
}


hkMemoryResourceContainer::hkMemoryResourceContainer( const char* name )
: hkResourceContainer(), m_name(name), m_parent(HK_NULL)
{
}

hkMemoryResourceContainer::hkMemoryResourceContainer(hkFinishLoadedObjectFlag flag) : hkResourceContainer(flag), m_name(flag), m_resourceHandles(flag), m_children(flag)
{
	if ( flag.m_finishing )
	{
		for( int i = 0; i < m_children.getSize(); ++i )
		{
			m_children[i]->m_parent = this;
		}
	}
}


hkMemoryResourceContainer::~hkMemoryResourceContainer()
{
}


hkResourceHandle* hkMemoryResourceContainer::createResource( const char* name, void* object, const hkClass* klass )
{
	hkMemoryResourceHandle* handle = new hkMemoryResourceHandle();
	handle->setName( name );
	handle->setObject( object, klass );
	m_resourceHandles.pushBack(handle);
	handle->removeReference();
	return handle;
}



hkResourceHandle* hkMemoryResourceContainer::findResourceByName( const char* objectName, const hkClass* klass, const hkResourceHandle* prevObject ) const
{
	int index = 0;
	while ((prevObject) && (index < m_resourceHandles.getSize()) && (m_resourceHandles[index++] != prevObject)) {}

	for (int i=index; i < m_resourceHandles.getSize(); i++)
	{
		hkResourceHandle* resourceHandle = m_resourceHandles[i];

		//
		//	Check name
		//
		if ( objectName )
		{
			hkStringBuf nameBuffer;
			if ( hkString::strCmp( objectName, resourceHandle->getName(nameBuffer) ) != 0 )
			{
				continue;
			}
		}

		//
		// check type
		//
		if ( klass )
		{
			const hkClass* linkedClass = resourceHandle->getClass();

			if ( ( klass != linkedClass ) && !klass->isSuperClass(*linkedClass)   )
			{
				if ( objectName )
				{
					HK_WARN_ALWAYS( 0xf034ed22, "Class mismatch, cannot resolve link: " << klass->getName() << " != " << linkedClass->getName() );
					return HK_NULL;
				}
				continue;
			}
		}

		return resourceHandle;
	}
	return HK_NULL;
}


void hkMemoryResourceContainer::destroyResource(hkResourceHandle* resourceHandle)
{
	hkMemoryResourceHandle* handle = static_cast<hkMemoryResourceHandle*>(resourceHandle);
	int index = m_resourceHandles.indexOf(handle);
	if ( index > -1 )
	{
		m_resourceHandles.removeAtAndCopy(index);
	}
}

const char* hkMemoryResourceContainer::getName( hkStringBuf& buffer ) const
{
	return m_name;
}

hkResourceContainer* hkMemoryResourceContainer::createContainer(const char* name)
{
	{
		hkResourceContainer* container = findContainerByName( name );
		if ( container )
		{
			return container;
		}
	}
	
	
	hkMemoryResourceContainer* container = new hkMemoryResourceContainer( name );
	m_children.pushBack(container);
	container->m_parent = this;
	container->removeReference();
	return container;
}

void hkMemoryResourceContainer::destroyContainer( hkResourceContainer* container2 )
{
	hkMemoryResourceContainer* container = static_cast<hkMemoryResourceContainer*>(container2);
	int index = m_children.indexOf(container);
	if ( index > -1 )
	{
		m_children.removeAt(index);
	}
}

int hkMemoryResourceContainer::getNumContainers()
{
	return m_children.getSize();
}

hkResourceContainer* hkMemoryResourceContainer::findContainerByName( const char* containerName, const hkResourceContainer* prevContainer  ) const
{
	int index = 0;
	while ((prevContainer) && (index < m_children.getSize()) && (m_children[index++] != prevContainer)) {}

	for (int i=index; i < m_children.getSize(); i++)
	{
		hkMemoryResourceContainer* container = m_children[i];

		//
		//	Check name
		//
		if ( containerName )
		{
			if ( hkString::strCmp( containerName, m_children[i]->m_name ) != 0)
			{
				continue;
			}
		}

		return container;
	}
	return HK_NULL;
}

hkResult hkMemoryResourceContainer::parentTo( hkResourceContainer* newParent )
{
	hkMemoryResourceContainer* newP = static_cast<hkMemoryResourceContainer*>( newParent );

	// check for circular dependency
	{
		for ( hkResourceContainer* p = newParent; p; p = p->getParent() )
		{
			if ( p == this )
			{
				HK_WARN_ALWAYS( 0xabba4554, "Cannot parent '" << m_name << "' to '" << newP->m_name << "' as this would create a circular dependency ");
				return HK_FAILURE;
			}
		}
	}


	
	// remove from old parent
	this->addReference();
	{
		int index = m_parent->m_children.indexOf( this );
		m_parent->m_children.removeAtAndCopy( index );
	}
	newP->m_children.pushBack( this );
	m_parent = newP;
	this->removeReference();
	return HK_SUCCESS;
}


hkContainerResourceMap::hkContainerResourceMap( class hkResourceContainer* container )
{
	hkArray<hkResourceHandle*> handles; container->findAllResourceRecursively( handles );

	for ( int i = 0; i < handles.getSize(); i++  )
	{
		hkResourceHandle* handle = handles[i];

		hkStringBuf buffer;
		const char* name = handle->getName( buffer );
		HK_ASSERT( 0xf032f612, name != buffer );
		m_resources.insert( name, handle );
	}
}

void* hkContainerResourceMap::findObjectByName( const char* objectName, const hkClass** klassOut ) const
{
	if ( klassOut )
	{
		*klassOut = HK_NULL;
	}

	hkResourceHandle* handle = m_resources.getWithDefault( objectName, HK_NULL);
	if ( !handle )
	{
		return HK_NULL;
	}
	if ( klassOut)
	{
		* klassOut = handle->getClass(); 
	}
	return handle->getObject();
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
