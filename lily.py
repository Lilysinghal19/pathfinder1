"""n=5
for i in range (1,n):
    for j in range (1,n):
        print(i,end=" ")
        
    print()    
*
**
***
****

for i in range(1,5):
    for j in range(i):
        print("*",end=" ")
    print()  
for i in range(1,5):
    for j in range(1,i+1):
        print(j,end=" ")
    print()   
ch=65
for i in range (1,5):
    for j in range (i):
        print(chr(ch),end=" ")
        ch+=1
    print()    
n=3
for i in range (1,n+1):
    print("*",end=" ")
    for j in range(1,n-1):
        if i==1 or i==n:
            print("*",end=" ")
        else:
            print (" ",end=" ")    
    print("*")  
n=5
num=1
for i in range(1,n):
    for i in range(i):
        print(num,end=" ")
        num+=1
    print()   
def multiply(a,b):
    result=0
    for i in range(b):
        result+=a
    print (result)    
multiply(4,5) 
def sum(n):
    return n + 1

def main():
    print(sum(5))

if __name__ == "__main__":
    main() """
"""def mul(a,b):
    if b==1:
        return a
    else:
        return a+mul(a,b-1)
print (mul(7,8))
number=int(input("enter the number:\t"))
if number%2==0:
    print("number is even")
else :
    print ("number is odd" )"""
"""===================================================================================================
class Lily:
    
    def __init__(self):
        self.pin=""
        self.balance=0
        
    
    
    def menu(self):
        
        user_input=int(input(how would you like to proceed
                         1.enter1 to enter a pin:
                         2.enter 2 to exit:
                         3.enter 3 to check balance:
                         4.enter 4 to withdraw:
                         5.enter 5 to deposit:\n)  )  
        if user_input==1:
            self.create_pin()
        elif user_input==2:
            print("exit")
        elif user_input==3:
            self.check_balance() 
        elif user_input==4:
            self.withdraw()
        elif user_input==5:
            self.deposit()
        else  :
            print("please enter the correct pin") 
    def create_pin(self):
        self.pin=int(input("enter your pin"))
        print("set pin successfully")
    def deposit(self):
        temp=int(input("enter your pin first"))
        if temp==self.pin:
            amount=int(input("enter the amount"))
            self.balance=self.balance+amount 
            print("deposit done successfully")   
        else:
            print("this is an invalid pin") 
    def withdraw(self):
        temp=int(input("enter your pin"))  
        if temp==self.pin:
            amount=int(input("enter the amount"))
            if amount<=self.balance:
                self.balance=self.balance-amount
                print(f"your withdrawl of {amount} is successfully done and now your balance is {self.balance}")
            else:
                print("insufficient funds")    
        else:
            print("you have entered an incorrect pin")        
    def check_balance (self):
        temp=int(input("enter your pin"))
        if temp==self.pin:
            print(f"your balance is{self.balance}")  
        else:
            print("you have entered an invalid pin")              
if __name__ == "__main__":
    lily = Lily()
n=int(input("enter a number"))
sum=0
for i in range(1,n+1):
    sum=sum+i
print(sum)"""
n=int(input("enter a number"))
for i in range(1,n+1):
    print(i)
   
        




     





      


    

        
    