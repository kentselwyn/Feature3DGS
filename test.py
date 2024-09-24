

def get_higher_msb(n):
    msb = 16  # Start from the highest bit position in a 32-bit number
    step = msb
    
    while step > 1:
        step //= 2  # Perform integer division to halve the step
        if n >> msb:  # Check if there is a 1 bit at or above the current MSB
            msb += step
        else:
            msb -= step
    
    # Final adjustment if needed
    if n >> msb:
        msb += 1
    
    return msb


# python test.py
if __name__=="__main__":
    n = (1<<0)
    print(get_higher_msb(n))