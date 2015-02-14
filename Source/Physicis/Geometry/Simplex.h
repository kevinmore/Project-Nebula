#pragma once
#include <Utility/EngineCommon.h>

class Simplex
{
public:
	Simplex();

	// Return true if the simplex contains 4 points
	inline bool isFull() const { return (m_curBits == 0xf); }

	// Return true if the simplex is empty
	inline bool isEmpty() const { return (m_curBits == 0x0); }

	// Return the maximum squared length of a point
	inline float maxLengthSqrd() const { return m_maxLengthSqrd; }

	int getPoints(QVector<vec3>& suppPointsA, QVector<vec3>& suppPointsB, QVector<vec3>& points) const;
	void addPoint(const vec3& point, const vec3& suppPointA, const vec3& suppPointB);
	bool isDegenerate(const vec3& point) const;
	bool isAffinelyIndependent() const;
	void closestPointAandB(vec3& pA, vec3& pB) const;
	bool runJohnsonAlgorithm(vec3& v);

private:
	// Return true if the bits of "b" is a subset of the bits of "a"
	inline bool isSubset(uint containerSet, uint subSet) const { return ((containerSet & subSet) == subSet); }

	bool isValidSubset(uint subset) const;
	void updateDiffLengths();
	void calcDeterminants();

	vec3 m_Points[4];
	float m_lengthSqrd[4]; // m_LengthSqrd[i] = m_Points[i].LengthSqrd()
	float m_maxLengthSqrd; // Maximum value among m_LengthSqrd[i]
	vec3 m_suppPointsA[4];
	vec3 m_suppPointsB[4];
	vec3 m_diffLength[4][4]; // m_DiffLength[i][j] = m_Points[i] - m_Points[j]
	float m_diffLengthSqrd[4][4];
	float m_det[16][4]; // Determinant 

	uint m_curBits;
	uint m_lastFound; // indicates a bit location found last time. It is between 0 to 3. 
	uint m_lastBit;   // m_LastBit = 1 << m_LastFound
	uint m_allBits;   // m_AllBits = m_CurBits | m_LastBit
};