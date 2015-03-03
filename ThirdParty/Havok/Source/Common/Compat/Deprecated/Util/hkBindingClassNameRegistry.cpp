/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/hkSerialize.h>
#include <Common/Compat/Deprecated/Util/hkBindingClassNameRegistry.h>
#include <Common/Base/Reflection/hkClassMemberAccessor.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Serialize/Util/hkStructureLayout.h>

static inline void computeMemberOffsetsInplace(const hkStringMap<hkClass*>& classes)
{
	hkStructureLayout layout;
	hkPointerMap<const hkClass*, int> done;
	hkStringMap<hkClass*>::Iterator iter = classes.getIterator();
	while( classes.isValid(iter) )
	{
		hkClass* klass = classes.getValue(iter);
		layout.computeMemberOffsetsInplace( *klass, done );
		iter = classes.getNext(iter);
	}
}

HK_COMPILE_TIME_ASSERT( sizeof(hkClassEnum::Item) == sizeof(hkInternalClassEnumItem) );
HK_COMPILE_TIME_ASSERT( sizeof(hkClassEnum) == sizeof(hkInternalClassEnum) );
HK_COMPILE_TIME_ASSERT( sizeof(hkClassMember) == sizeof(hkInternalClassMember) );

hkBindingClassNameRegistry::ClassAllocationsTracker::UnresolvedClassPointerTracker::UnresolvedClassPointerTracker() {}
hkBindingClassNameRegistry::ClassAllocationsTracker::UnresolvedClassPointerTracker::~UnresolvedClassPointerTracker()
{
	HK_ASSERT(0x15a09060, m_pointers.getNumKeys() == 0);
}

void hkBindingClassNameRegistry::ClassAllocationsTracker::UnresolvedClassPointerTracker::classPointedBy( const hkClass* newClass, const hkClass** fromWhere )
{
	//HK_SERIALIZE_LOG(("TrackObjectPointedBy(obj=0x%p,loc=0x%p)\n", newClass, fromWhere));
	const hkClass* oldClass = *fromWhere;
	if (oldClass)
	{
		for( int i = m_pointers.getFirstIndex(const_cast<hkClass*>(oldClass));
			i != -1;
			i = m_pointers.getNextIndex(i) )
		{
			if( m_pointers.getValue(i) == fromWhere )
			{
				if( newClass == oldClass )
				{
					return;
				}
				m_pointers.removeByIndex(const_cast<hkClass*>(oldClass), i);
				break;
			}
		}
	}
	if( newClass )
	{
		m_pointers.insert( const_cast<hkClass*>(newClass), fromWhere );
	}
	*fromWhere = newClass;
}

// replace class entry and remove it from tracking
void hkBindingClassNameRegistry::ClassAllocationsTracker::UnresolvedClassPointerTracker::replaceClassAndRemove( const hkClass* oldClass, const hkClass* newClass )
{
	//HK_SERIALIZE_LOG(("TrackReplaceObject(oldObj=0x%p,newObj=0x%p)\n", oldClass, newClass));
	// replace pointers to old object with pointers to new one
	int index = m_pointers.getFirstIndex(const_cast<hkClass*>(oldClass));
	while( index != -1 )
	{
		const hkClass** ptrOldClass = m_pointers.getValue(index);
		HK_ASSERT(0x15a09066, *ptrOldClass == oldClass );
		*ptrOldClass = newClass;
		index = m_pointers.getNextIndex(index);
	}
	m_pointers.removeKey( const_cast<hkClass*>(oldClass) );
}

void hkBindingClassNameRegistry::ClassAllocationsTracker::UnresolvedClassPointerTracker::resolveClasses(hkPointerMap<const hkClass*, const hkClass*>& classFromUnresolved)
{
	hkPointerMap<hkClass*, hkClass*>::Iterator iter = classFromUnresolved.getIterator();
	while( classFromUnresolved.isValid(iter) )
	{
		replaceClassAndRemove(classFromUnresolved.getKey(iter), classFromUnresolved.getValue(iter));
		iter = classFromUnresolved.getNext(iter);
	}
}

int hkBindingClassNameRegistry::ClassAllocationsTracker::UnresolvedClassPointerTracker::getSize() const
{
	return m_pointers.getNumKeys();
}

hkBindingClassNameRegistry::ClassAllocationsTracker::ClassAllocationsTracker() { }

hkBindingClassNameRegistry::ClassAllocationsTracker::~ClassAllocationsTracker()
{
	for( int i = 0; i < m_trackedClassData.getSize(); ++i )
	{
		deallocateClass(m_trackedClassData[i].allocatedClass, m_trackedClassData[i].enumsAllocatedForMembers);
	}
}

hkClass* hkBindingClassNameRegistry::ClassAllocationsTracker::restoreClassHierarchy(const hkClass& classToCopy, const hkBindingClassNameRegistry* classRegistry, const hkStringMap<const char*>& oldNameFromNewName, hkStringMap<hkClass*>& classesInOut )
{
	UnresolvedClassPointerTracker unresolvedPointersTracker;

	hkClass* allocatedClass = allocateClass(classToCopy, oldNameFromNewName.getWithDefault(classToCopy.getName(), classToCopy.getName()), unresolvedPointersTracker);
	classesInOut.insert(allocatedClass->getName(), allocatedClass);
	// resolve references to existing classes
	if( unresolvedPointersTracker.getSize() > 0 )
	{
		hkPointerMap<const hkClass*, const hkClass*> availableClassFromUnresolved;
		{
			hkPointerMap<hkClass*, int>::Iterator iter = unresolvedPointersTracker.m_pointers.m_indexMap.getIterator();
			while( unresolvedPointersTracker.m_pointers.m_indexMap.isValid(iter) )
			{
				const hkClass* unresolvedClass = (const hkClass*)(unresolvedPointersTracker.m_pointers.m_indexMap.getKey(iter));
				const char* className = oldNameFromNewName.getWithDefault(unresolvedClass->getName(), unresolvedClass->getName());
				const hkClass* availableClass = classRegistry->getClassByNameNoRecurse(className);
				if( !availableClass )
				{
					availableClass = classesInOut.getWithDefault(className, HK_NULL);
				}
				if( availableClass )
				{
					availableClassFromUnresolved.insert(unresolvedClass, availableClass);
				}
				iter = unresolvedPointersTracker.m_pointers.m_indexMap.getNext(iter);
			}
		}
		unresolvedPointersTracker.resolveClasses(availableClassFromUnresolved);
	}
	// rebuild missing classes and resolve references
	if( unresolvedPointersTracker.getSize() > 0 )
	{
		hkPointerMap<const hkClass*, const hkClass*> availableClassFromUnresolved;
		hkPointerMap<hkClass*, int>::Iterator iter = unresolvedPointersTracker.m_pointers.m_indexMap.getIterator();
		while( unresolvedPointersTracker.m_pointers.m_indexMap.isValid(iter) )
		{
			const hkClass* unresolvedClass = (const hkClass*)(unresolvedPointersTracker.m_pointers.m_indexMap.getKey(iter));
			hkClass* availableClass = restoreClassHierarchy(*unresolvedClass, classRegistry, oldNameFromNewName, classesInOut);
			availableClassFromUnresolved.insert(unresolvedClass, availableClass);
			iter = unresolvedPointersTracker.m_pointers.m_indexMap.getNext(iter);
		}
		unresolvedPointersTracker.resolveClasses(availableClassFromUnresolved);
	}
	return allocatedClass;
}

hkInternalClassEnum* hkBindingClassNameRegistry::ClassAllocationsTracker::allocateEnums(const hkInternalClassEnum* enumsToCopy, int numEnums)
{
	if( !enumsToCopy )
	{
		return HK_NULL;
	}
	hkInternalClassEnum* allocatedEnums = hkAllocate<hkInternalClassEnum>(numEnums, HK_MEMORY_CLASS_SERIALIZE);
	hkString::memCpy(allocatedEnums, enumsToCopy, sizeof(hkInternalClassEnum)*numEnums);
	for( int i = 0; i < numEnums; ++i )
	{
		allocatedEnums[i].m_name = hkString::strDup(allocatedEnums[i].m_name);
		allocatedEnums[i].m_attributes = HK_NULL;
		allocatedEnums[i].m_items = hkAllocate<hkInternalClassEnumItem>(allocatedEnums[i].m_numItems, HK_MEMORY_CLASS_SERIALIZE);
		hkString::memCpy(const_cast<hkInternalClassEnumItem*>(allocatedEnums[i].m_items), enumsToCopy[i].m_items, sizeof(hkInternalClassEnumItem)*allocatedEnums[i].m_numItems);
		for( int j = 0; j < allocatedEnums[i].m_numItems; ++j )
		{
			const_cast<hkInternalClassEnumItem*>(allocatedEnums[i].m_items)[j].m_name = hkString::strDup(allocatedEnums[i].m_items[j].m_name);
		}
	}
	return allocatedEnums;
}

hkInternalClassMember* hkBindingClassNameRegistry::ClassAllocationsTracker::allocateMembers(const hkInternalClassMember* membersToCopy, int numMembers,
																  const hkInternalClassEnum* originalEnums, const hkInternalClassEnum* copiedEnums, int numAvailableEnums,
																  hkPointerMap<hkInternalClassEnum*, hkBool32>& enumsAllocatedForMembersOut, UnresolvedClassPointerTracker& unresolvedClassPointersInOut)
{
	if( !membersToCopy )
	{
		return HK_NULL;
	}
	hkInternalClassMember* allocatedMembers = hkAllocate<hkInternalClassMember>(numMembers, HK_MEMORY_CLASS_SERIALIZE);
	hkString::memCpy(allocatedMembers, membersToCopy, sizeof(hkInternalClassMember)*numMembers);
	hkPointerMap<const hkClassEnum*, hkInternalClassEnum*> allocatedEnumFromOriginal;
	for( int i = 0; i < numMembers; ++i )
	{
		allocatedMembers[i].m_name = hkString::strDup(allocatedMembers[i].m_name);
		allocatedMembers[i].m_attributes = HK_NULL;
		if( allocatedMembers[i].m_class )
		{
			if( allocatedMembers[i].m_class->getFlags().allAreSet(hkClass::FLAGS_NOT_SERIALIZABLE) )
			{
				allocatedMembers[i].m_class = HK_NULL;
			}
			else
			{
				unresolvedClassPointersInOut.classPointedBy(allocatedMembers[i].m_class, &allocatedMembers[i].m_class);
			}
		}
		else if( allocatedMembers[i].m_enum )
		{
			for( int j = 0; j < numAvailableEnums; ++j )
			{
				if( allocatedMembers[i].m_enum == reinterpret_cast<const hkClassEnum*>(&originalEnums[j]) )
				{
					allocatedMembers[i].m_enum = reinterpret_cast<const hkClassEnum*>(&copiedEnums[j]);
					enumsAllocatedForMembersOut.insert(reinterpret_cast<hkInternalClassEnum*>(const_cast<hkClassEnum*>(allocatedMembers[i].m_enum)), false);
					break;
				}
			}
			if( !enumsAllocatedForMembersOut.hasKey(reinterpret_cast<hkInternalClassEnum*>(const_cast<hkClassEnum*>(allocatedMembers[i].m_enum))) )
			{
				// the enum pointer is still original at this stage
				hkInternalClassEnum* allocatedEnum = HK_NULL;
				if( allocatedEnumFromOriginal.hasKey(allocatedMembers[i].m_enum) )
				{
					allocatedEnumFromOriginal.get(allocatedMembers[i].m_enum, &allocatedEnum);
				}
				else
				{
					allocatedEnum = allocateEnums(reinterpret_cast<const hkInternalClassEnum*>(allocatedMembers[i].m_enum), 1);
					enumsAllocatedForMembersOut.insert(allocatedEnum, true);
					allocatedEnumFromOriginal.insert(allocatedMembers[i].m_enum, allocatedEnum);
				}
				allocatedMembers[i].m_enum = reinterpret_cast<hkClassEnum*>(allocatedEnum);
			}
		}
	}
	return allocatedMembers;
}

void* hkBindingClassNameRegistry::ClassAllocationsTracker::allocateDefaults(const hkClass& klass, void* defaultsToCopy)
{
	if( !defaultsToCopy )
	{
		return HK_NULL;
	}
	// calc size of defaults
	hkInt32 defaultsSize = 0;
	hkInt32* offsets = static_cast<hkInt32*>(defaultsToCopy);
	// find the last member offset
	for( int i = klass.getNumDeclaredMembers()-1; i >= 0; --i )
	{
		if( offsets[i] >=0 )
		{
			const hkClassMember& mem = klass.getDeclaredMember(i);
			// found last member offset
			defaultsSize = offsets[i] + mem.getSizeInBytes();
			break;
		}
	}

	if (defaultsSize==0)
	{
		return HK_NULL;
	}

	HK_ASSERT(0x15a09061, defaultsSize > 0);
	char* allocatedDefaults = hkAllocate<char>(defaultsSize, HK_MEMORY_CLASS_SERIALIZE);
	hkString::memCpy(allocatedDefaults, defaultsToCopy, defaultsSize);
	return allocatedDefaults;
}

hkClass* hkBindingClassNameRegistry::ClassAllocationsTracker::allocateClass(const hkClass& classToDuplicate, const char* classNameToSet, UnresolvedClassPointerTracker& unresolvedClassPointersOut)
{
	ClassData* classData = m_trackedClassData.expandBy(1);
	classData->allocatedClass = hkAllocate<hkClass>(1, HK_MEMORY_CLASS_SERIALIZE);
	HK_ASSERT(0x15a09062, classData->allocatedClass);
	if( classToDuplicate.getParent() )
	{
		hkClassAccessor klass(classData->allocatedClass, &hkClassClass);
		unresolvedClassPointersOut.classPointedBy(classToDuplicate.getParent(), static_cast<const hkClass**>(klass.member("parent").getAddress()));
	}
	hkClassAccessor klass(const_cast<hkClass*>(&classToDuplicate), &hkClassClass);
	// name
	char* className = classNameToSet ? hkString::strDup(classNameToSet) : hkString::strDup(classToDuplicate.getName());
	// enums
	hkClassMemberAccessor::SimpleArray& declaredEnums = klass.member("declaredEnums").asSimpleArray();
	hkInternalClassEnum* allocatedEnums = allocateEnums(static_cast<hkInternalClassEnum*>(declaredEnums.data), declaredEnums.size);
	// members
	hkClassMemberAccessor::SimpleArray& declaredMembers = klass.member("declaredMembers").asSimpleArray();
	hkInternalClassMember* allocatedMembers = allocateMembers(static_cast<hkInternalClassMember*>(declaredMembers.data), declaredMembers.size,
		static_cast<hkInternalClassEnum*>(declaredEnums.data), allocatedEnums, declaredEnums.size, classData->enumsAllocatedForMembers, unresolvedClassPointersOut);
	// defaults
	char* defaults = static_cast<char*>(allocateDefaults(classToDuplicate, klass.member("defaults").asPointer()));
	// init the class content
	new (classData->allocatedClass) hkClass(className, classToDuplicate.getParent(), 0,
		HK_NULL, klass.member("numImplementedInterfaces").asInt32(),
		reinterpret_cast<hkClassEnum*>(allocatedEnums), declaredEnums.size,
		reinterpret_cast<hkClassMember*>(allocatedMembers), declaredMembers.size,
		defaults,
		HK_NULL, klass.member("flags").asUint32());
	return classData->allocatedClass;
}

void hkBindingClassNameRegistry::ClassAllocationsTracker::deallocateEnums(hkInternalClassEnum* enums, int numEnums)
{
#	if defined(HK_DEBUG)
	if( numEnums > 0 )
	{
		HK_ASSERT(0x15a09063, enums);
	}
#	endif
	for( int i = 0; i < numEnums; ++i )
	{
		// deallocate enum items
		hkInternalClassEnum& classEnum = enums[i];
		hkInternalClassEnumItem* classEnumItems = const_cast<hkInternalClassEnumItem*>(classEnum.m_items);
		HK_ASSERT(0x15a09064, classEnumItems);
		for( int k = 0; k < classEnum.m_numItems; ++k )
		{
			hkDeallocate<char>(const_cast<char*>(classEnumItems[k].m_name));
		}
		hkDeallocate<hkInternalClassEnumItem>(classEnumItems);
		hkDeallocate<char>(const_cast<char*>(classEnum.m_name));
	}
	hkDeallocate<hkInternalClassEnum>(enums);
}

// deallocate members
void hkBindingClassNameRegistry::ClassAllocationsTracker::deallocateMembers(hkInternalClassMember* members, int numMembers, hkPointerMap<hkInternalClassEnum*, hkBool32>& enumsAllocatedForMembersInOut)
{
#		if defined(HK_DEBUG)
	if( numMembers > 0 )
	{
		HK_ASSERT(0x15a09065, members);
	}
#		endif
	for( int i = 0; i < numMembers; ++i )
	{
		// deallocate enum items
		hkInternalClassMember& classMember = members[i];
		if( classMember.m_enum )
		{
			hkInternalClassEnum* memberEnum = static_cast<hkInternalClassEnum*>(static_cast<void*>(const_cast<hkClassEnum*>(classMember.m_enum)));
			if( enumsAllocatedForMembersInOut.hasKey(memberEnum)
				&& enumsAllocatedForMembersInOut.getWithDefault(memberEnum, false) )
			{
				deallocateEnums(memberEnum, 1);
				enumsAllocatedForMembersInOut.insert(memberEnum, false);
			}
		}
		hkDeallocate<char>(const_cast<char*>(classMember.m_name));
	}
	hkDeallocate<hkInternalClassMember>(members);
}

void hkBindingClassNameRegistry::ClassAllocationsTracker::deallocateClass(hkClass* classToDeallocate, hkPointerMap<hkInternalClassEnum*, hkBool32>& enumsAllocatedForMembersInOut)
{
	hkClassAccessor klass(classToDeallocate, &hkClassClass);
	// deallocate enums
	hkClassMemberAccessor::SimpleArray& declaredEnums = klass.member("declaredEnums").asSimpleArray();
	deallocateEnums(static_cast<hkInternalClassEnum*>(declaredEnums.data), declaredEnums.size);
	// deallocate members
	hkClassMemberAccessor::SimpleArray& declaredMembers = klass.member("declaredMembers").asSimpleArray();
	deallocateMembers(static_cast<hkInternalClassMember*>(declaredMembers.data), declaredMembers.size, enumsAllocatedForMembersInOut);
	// deallocate defaults
	hkDeallocate<char>(static_cast<char*>(klass.member("defaults").asPointer()));
	// deallocate class
	hkDeallocate<char>(klass.member("name").asCstring());
	hkDeallocate<hkClass>(classToDeallocate);
}

hkBindingClassNameRegistry::hkBindingClassNameRegistry(const hkVersionRegistry::ClassRename* renames, const hkClassNameRegistry* next) :
hkChainedClassNameRegistry(next)
{
	if( renames )
	{
		for( int i = 0; renames[i].oldName; ++i )
		{
			m_newNameFromOldName.insert(renames[i].oldName, renames[i].newName);
			m_oldNameFromNewName.insert(renames[i].newName, renames[i].oldName);
		}
	}
}

hkBindingClassNameRegistry::~hkBindingClassNameRegistry()
{
}

const hkClass* hkBindingClassNameRegistry::getClassByNameNoRecurse( const char* className ) const
{
	const hkClassNameRegistry* savedReg = m_nextRegistry;
	const_cast<hkBindingClassNameRegistry*>(this)->m_nextRegistry = HK_NULL;
	const hkClass* klass = hkChainedClassNameRegistry::getClassByName( className );
	const_cast<hkBindingClassNameRegistry*>(this)->m_nextRegistry = savedReg;
	return klass;
}

/// Get a class by name or HK_NULL if it was not registered.
const hkClass* hkBindingClassNameRegistry::getClassByName( const char* className ) const
{
	const hkClass* klass = getClassByNameNoRecurse( className );
	if( klass || !m_nextRegistry )
	{
		return klass;
	}
	// try to restore class
	// check renames
	const char* realName = getNewName(className);
	klass = hkChainedClassNameRegistry::getClassByName( realName );
	if( klass && !klass->getFlags().allAreSet(hkClass::FLAGS_NOT_SERIALIZABLE) )
	{
		// duplicate the class found in the next registry
		// assume that the class is ready to use
		hkStringMap<hkClass*> classes;
		const hkClass* allocatedClass = m_tracker.restoreClassHierarchy(*klass, this, m_oldNameFromNewName, classes);
		computeMemberOffsetsInplace(classes);
		const_cast<hkBindingClassNameRegistry*>(this)->merge(reinterpret_cast<hkStringMap<const hkClass*>&>(classes));
		return allocatedClass;
	}
	return HK_NULL;
}

const char* hkBindingClassNameRegistry::getNewName(const char* oldName) const
{
	return m_newNameFromOldName.getWithDefault(oldName, oldName);
}

void hkBindingClassNameRegistry::registerRenames(const hkVersionRegistry::ClassRename* renames)
{
	if( renames )
	{
		for( int i = 0; renames[i].oldName; ++i )
		{
			m_newNameFromOldName.insert(renames[i].oldName, renames[i].newName);
			m_oldNameFromNewName.insert(renames[i].newName, renames[i].oldName);
		}
	}
}

void hkBindingClassNameRegistry::registerRenames(const hkStringMap<const char*>& newNameFromOldNameMap)
{
	hkStringMap<const char*>::Iterator iter = newNameFromOldNameMap.getIterator();
	while( newNameFromOldNameMap.isValid(iter) )
	{
		const char* oldName = newNameFromOldNameMap.getKey(iter);
		const char* newName = newNameFromOldNameMap.getValue(iter);
		m_newNameFromOldName.insert(oldName, newName);
		m_oldNameFromNewName.insert(newName, oldName);
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
