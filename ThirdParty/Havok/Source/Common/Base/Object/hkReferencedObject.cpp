/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Reflection/hkTypeInfo.h>

static void HK_CALL finishLoadedObjecthkBaseObject(void*, int)
{
}
static void HK_CALL cleanupLoadedObjecthkBaseObject(void*)
{
}
static const void* HK_CALL getVtablehkBaseObject()
{
#if HK_LINKONCE_VTABLES==1
	hkFinishLoadedObjectFlag loadedFlag;
	hkBaseObject o(loadedFlag);
	return *(void**)(&o);
#else
	return HK_VTABLE_FROM_CLASS(hkBaseObject);
#endif
}
extern const hkTypeInfo hkBaseObjectTypeInfo;
const hkTypeInfo hkBaseObjectTypeInfo( "hkBaseObject", "!hkBaseObject", finishLoadedObjecthkBaseObject, cleanupLoadedObjecthkBaseObject, getVtablehkBaseObject(), sizeof(hkReferencedObject));

extern void HK_CALL hkReferenceCountError(const hkReferencedObject*,const char*);

#if !defined(HK_PLATFORM_SPU)
const hkClass* hkReferencedObject::getClassType() const
{
    return HK_NULL;
}
#endif //!defined(HK_PLATFORM_SPU)

#include <Common/Base/DebugUtil/MultiThreadCheck/hkMultiThreadCheck.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>

#define LOCK_MAGIC 0x23df4554

class hkReferencedObjectLock : public hkReferencedObject, public hkSingleton<hkReferencedObjectLock>
{
	public:
		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE);
		typedef hkUint32 ThreadId;
		hkReferencedObjectLock() : m_criticalSection(4000)
		{
			m_lockMode = LOCK_MODE_AUTO;
#if !defined(HK_PLATFORM_SPU)
			m_multiThreadCheck.enableChecks();
#endif
			m_lockCount = 0;
		}

		HK_FORCE_INLINE void lock()
		{
			hkUint32* lockedAllPtr = hkMemoryRouter::getInstance().getRefObjectLocalStore();
			if ( *lockedAllPtr == LOCK_MAGIC )
			{
				m_lockCount++;
				return;
			}
			m_criticalSection.enter();
			m_lockCount = 1;
			*lockedAllPtr = LOCK_MAGIC;
#if defined(HK_DEBUG) && !defined(HK_PLATFORM_SPU)
			m_multiThreadCheck.markForWrite();
#endif
		}

		HK_FORCE_INLINE void unlock()
		{
			hkUint32* lockedAllPtr = hkMemoryRouter::getInstance().getRefObjectLocalStore();
			HK_ASSERT2( 0xf0342123, *lockedAllPtr == LOCK_MAGIC, "Unmatched unlock");
			if (--m_lockCount > 0)
			{
				return;
			}
#if defined(HK_DEBUG) && !defined(HK_PLATFORM_SPU)
			m_multiThreadCheck.unmarkForWrite();
#endif
			*lockedAllPtr = 0;
			m_criticalSection.leave();
		}

	public:
		hkReferencedObject::LockMode m_lockMode;
		int m_lockCount;

		hkMultiThreadCheck m_multiThreadCheck;
		hkCriticalSection  m_criticalSection;
};

HK_SINGLETON_MANUAL_IMPLEMENTATION(hkReferencedObjectLock);

void hkReferencedObject::initializeLock()
{
	hkReferencedObjectLock::replaceInstance(new hkReferencedObjectLock());
}

void hkReferencedObject::deinitializeLock()
{
	hkReferencedObjectLock::replaceInstance(HK_NULL);
}

void hkReferencedObject::setLockMode(LockMode mode)
{
	hkReferencedObjectLock* lock = &hkReferencedObjectLock::getInstance();
	lock->m_lockMode = mode;
}

void hkReferencedObject::lockInit(LockMode lockMode)
{
	hkReferencedObjectLock* lock = &hkReferencedObjectLock::getInstance();
	if ( !lock )
	{
		lock = new hkReferencedObjectLock();
		lock->m_lockMode = lockMode;
		hkReferencedObjectLock::replaceInstance(lock);
	}
	else
	{
		lock->m_lockMode = lockMode;
	}
}

hkReferencedObject::LockMode hkReferencedObject::getLockMode()
{
	hkReferencedObjectLock* lock = &hkReferencedObjectLock::getInstance();
	return lock->m_lockMode;
}

void hkReferencedObject::lockAll()
{
	hkReferencedObjectLock* lock = &hkReferencedObjectLock::getInstance();
	lock->lock();
}

void hkReferencedObject::unlockAll()
{
	hkReferencedObjectLock* lock = &hkReferencedObjectLock::getInstance();
	lock->unlock();
}

#if defined(HK_DEBUG)
	#define CHECK_INVALID_REFCOUNT(fromWhere) \
		if( m_referenceCount <= 0 ) \
		{ \
			hkReferenceCountError(this, fromWhere); \
		}
#else
	#define CHECK_INVALID_REFCOUNT(whereFrom)
#endif

void hkReferencedObject::addReference() const
{
	// we don't bother keeping references if the reference is going to be ignored.
	if ( m_memSizeAndFlags != 0 )
	{
		hkReferencedObjectLock* lock = &hkReferencedObjectLock::getInstance();
#if defined(HK_DEBUG)
		if ( lock->m_lockMode == hkReferencedObject::LOCK_MODE_MANUAL )
		{
			HK_ACCESS_CHECK_OBJECT( (&hkReferencedObjectLock::getInstance()), HK_ACCESS_RW );
		}
		CHECK_INVALID_REFCOUNT("addReference");
#endif
		const hkUint32* lockedAllPtr = hkMemoryRouter::getInstance().getRefObjectLocalStore();
		if ( (lock->m_lockMode == LOCK_MODE_AUTO) &&  (*lockedAllPtr != LOCK_MAGIC) )
		{
			lock->lock();
			++m_referenceCount;
			lock->unlock();
		}
		else // mode none | mode manual | mode auto with lock already acquired
		{
			++m_referenceCount;
		}
	}
}

void hkReferencedObject::removeReference() const
{
	if ( m_memSizeAndFlags == 0 )	
	{
		// not refcounted because data within packfile
		return;
	}

	// if the reference count == 1, we must be the only owner to 
	// this object, so there can't be any other thread touching this
	// reference.
	// Note: the way Havok shuts down, it relies on this behavior.
	if ( m_referenceCount == 1)
	{
		m_referenceCount = 0;
		delete this;
		return;
	}

	hkReferencedObjectLock* lock = &hkReferencedObjectLock::getInstance();
	// This should never brake unless the hkReferencedObjectLock singleton was prematurely deleted.
	HK_ON_DEBUG( if(!lock) HK_BREAKPOINT(0); )
#if defined(HK_DEBUG)
	if ( lock->m_lockMode == hkReferencedObject::LOCK_MODE_MANUAL )
	{
		HK_ACCESS_CHECK_OBJECT( (&hkReferencedObjectLock::getInstance()), HK_ACCESS_RW );
	}
#endif
	int oldRefCounter;
	const hkUint32* lockedAllPtr = hkMemoryRouter::getInstance().getRefObjectLocalStore();
	if ( (lock->m_lockMode == LOCK_MODE_AUTO) &&  (*lockedAllPtr != LOCK_MAGIC ) )
	{
		lock->lock();
		CHECK_INVALID_REFCOUNT("removeReference");
		oldRefCounter = m_referenceCount;
		m_referenceCount = hkInt16(oldRefCounter-1);
		lock->unlock();
	}
	else // mode none | mode manual | mode auto with lock already acquired
	{
		CHECK_INVALID_REFCOUNT("removeReference");
		oldRefCounter = m_referenceCount;
		m_referenceCount = hkInt16(oldRefCounter-1);
	}
	if ( oldRefCounter == 1 )
	{
		delete this;
	}
}


void hkReferencedObject::addReferenceLockUnchecked() const
{
	// we don't bother keeping references if the reference is going to be ignored.
	if ( m_memSizeAndFlags != 0 )
	{
		CHECK_INVALID_REFCOUNT("addReferenceUnchecked");
		++m_referenceCount;
	}
}

void hkReferencedObject::removeReferenceLockUnchecked() const
{
	if ( m_memSizeAndFlags != 0 )
	{
		CHECK_INVALID_REFCOUNT("removeReferenceUnchecked");
		--m_referenceCount;

		if ( m_referenceCount == 0 )
		{
			delete this;
		}
	}
}

void hkReferencedObject::addReferences( const hkReferencedObject*const* objects, int numObjects, int pointerStriding )
{
	hkReferencedObjectLock* lock = &hkReferencedObjectLock::getInstance();
	const hkUint32* lockedAllPtr = hkMemoryRouter::getInstance().getRefObjectLocalStore();
	if ( (lock->m_lockMode != LOCK_MODE_NONE) &&  (*lockedAllPtr != LOCK_MAGIC) )
	{
		lock->lock();
		for (int i = 0; i < numObjects; i++)
		{
			objects[0]->addReferenceLockUnchecked();
			objects = hkAddByteOffsetConst(objects, pointerStriding);
		}
		lock->unlock();
	}
	else
	{
		for (int i = 0; i < numObjects; i++)
		{
			objects[0]->addReferenceLockUnchecked();
			objects = hkAddByteOffsetConst(objects, pointerStriding);
		}
	}
}

void hkReferencedObject::removeReferences( const hkReferencedObject*const* objects, int numObjects, int pointerStriding )
{
	hkReferencedObjectLock* lock = &hkReferencedObjectLock::getInstance();
	const hkUint32* lockedAllPtr = hkMemoryRouter::getInstance().getRefObjectLocalStore();
	if ( (lock->m_lockMode != LOCK_MODE_NONE) &&  (*lockedAllPtr != LOCK_MAGIC ) )
	{
		lock->lock();
		for (int i = 0; i < numObjects; i++)
		{
			objects[0]->removeReferenceLockUnchecked();
			objects = hkAddByteOffsetConst(objects, pointerStriding);
		}
		lock->unlock();
	}
	else
	{
		for (int i = 0; i < numObjects; i++)
		{
			objects[0]->removeReferenceLockUnchecked();
			objects = hkAddByteOffsetConst(objects, pointerStriding);
		}
	}
}




hkCriticalSection* hkReferencedObject::getLockCriticalSection()
{
	hkReferencedObjectLock* lock = &hkReferencedObjectLock::getInstance();
	return &lock->m_criticalSection;
}


/*
HK_FORCE_INLINE hkReferencedObject::~hkReferencedObject()
{
#ifdef HK_DEBUG
extern void HK_CALL hkRemoveReferenceError(const hkReferencedObject*, const char*);

// if we are in a dtor and the do not delete flag
// is set then we have call delete on an object that does not own its memory and
// the dtor should never have been called.
// But objects which are not new'd will all have random values in this size
// param and we can't set it in the ctor to something other than 0 as the
// the ctor is called after the alloc in hkMemory (where the size mem is set..)
// if ( m_memSizeAndFlags == 0 ) ..

// reference count is either zero because this method was called
// from removeReference or one because we are a local variable.
if( (m_referenceCount & (~1)) != 0)
{
hkRemoveReferenceError(this,"hkReferencedObject destructor");
}
// After calling delete the reference count should always be zero. This catches
// cases where delete is explicitly called.
m_referenceCount = 0;
#endif
}
*/

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
